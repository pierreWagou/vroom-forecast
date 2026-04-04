"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Car,
  DollarSign,
  Image,
  Loader2,
  Monitor,
  RefreshCw,
  Save,
  Settings,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  predict,
  predictById,
  fetchHealth,
  benchmark,
  benchmarkById,
  reloadModel,
  saveVehicle,
  listVehicles,
  fetchVehicleFeatures,
  type VehicleFeatures,
  type PredictionResponse,
  type HealthResponse,
  type BenchmarkResponse,
  type VehicleRecord,
  type ComputedFeatures,
} from "@/lib/api";

const DEFAULT_VEHICLE: VehicleFeatures = {
  technology: 1,
  actual_price: 45,
  recommended_price: 50,
  num_images: 8,
  street_parked: 0,
  description: 250,
};

function StatusDot({ online }: { online: boolean }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${online ? "bg-turo-teal animate-pulse" : "bg-destructive"}`}
    />
  );
}

export default function Home() {
  const [vehicle, setVehicle] = useState<VehicleFeatures>(DEFAULT_VEHICLE);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [benchmarkResult, setBenchmarkResult] =
    useState<BenchmarkResponse | null>(null);
  const [benchmarkStoreResult, setBenchmarkStoreResult] =
    useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [benchLoading, setBenchLoading] = useState(false);
  const [benchStoreLoading, setBenchStoreLoading] = useState(false);
  const [benchProgress, setBenchProgress] = useState(0);
  const [benchStoreProgress, setBenchStoreProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedVehicles, setSavedVehicles] = useState<VehicleRecord[]>([]);
  const [vehiclePredictions, setVehiclePredictions] = useState<
    Record<number, PredictionResponse>
  >({});
  const [vehicleFeatures, setVehicleFeatures] = useState<
    Record<number, ComputedFeatures>
  >({});

  useEffect(() => {
    setMounted(true);
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const handleReload = async () => {
    setReloading(true);
    setError(null);
    try {
      const result = await reloadModel();
      // Refresh health to get updated version
      const h = await fetchHealth();
      setHealth(h);
      if (result.status === "reloaded") {
        setError(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reload failed");
    } finally {
      setReloading(false);
    }
  };

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await predict(vehicle);
      setPrediction(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [vehicle]);

  const handleBenchmark = async () => {
    setBenchLoading(true);
    setBenchProgress(0);
    setError(null);
    const timer = setInterval(
      () => setBenchProgress((p) => Math.min(p + 8, 90)),
      200
    );
    try {
      const result = await benchmark(vehicle, 1000);
      setBenchmarkResult(result);
      setBenchProgress(100);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Benchmark failed");
    } finally {
      clearInterval(timer);
      setBenchLoading(false);
    }
  };

  const handleBenchmarkStore = async () => {
    // Use the first materialized vehicle from the fleet
    const materialized = savedVehicles.find(
      (v) => vehicleFeatures[v.vehicle_id]?.materialized
    );
    if (!materialized) {
      setError("No materialized vehicle found. Save a vehicle first and wait for materialization.");
      return;
    }
    setBenchStoreLoading(true);
    setBenchStoreProgress(0);
    setError(null);
    const timer = setInterval(
      () => setBenchStoreProgress((p) => Math.min(p + 8, 90)),
      200
    );
    try {
      const result = await benchmarkById(materialized.vehicle_id, 1000);
      setBenchmarkStoreResult(result);
      setBenchStoreProgress(100);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Online store benchmark failed");
    } finally {
      clearInterval(timer);
      setBenchStoreLoading(false);
    }
  };

  const update = (field: keyof VehicleFeatures, value: number) =>
    setVehicle((prev) => ({ ...prev, [field]: value }));

  const handleSaveVehicle = async () => {
    setSaving(true);
    setError(null);
    try {
      const result = await saveVehicle(vehicle);
      const vehicles = await listVehicles();
      setSavedVehicles(vehicles);
      // Mark as pending — the polling effect will update when materialized
      setVehicleFeatures((prev) => ({
        ...prev,
        [result.vehicle_id]: { vehicle_id: result.vehicle_id, materialized: false } as ComputedFeatures,
      }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const handlePredictById = async (vehicleId: number) => {
    setError(null);
    try {
      const result = await predictById(vehicleId);
      setVehiclePredictions((prev) => ({ ...prev, [vehicleId]: result }));
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message.includes("500")
            ? `Vehicle #${vehicleId} not yet materialized. Run the feature pipeline first.`
            : e.message
          : "Prediction failed"
      );
    }
  };

  const handlePredictAll = async () => {
    setError(null);
    for (const v of savedVehicles) {
      try {
        const result = await predictById(v.vehicle_id);
        setVehiclePredictions((prev) => ({ ...prev, [v.vehicle_id]: result }));
      } catch {
        // Skip vehicles not yet materialized
      }
    }
  };

  const refreshVehicles = useCallback(async () => {
    try {
      const vehicles = await listVehicles();
      setSavedVehicles(vehicles);
      // Fetch computed features for each vehicle
      const features: Record<number, ComputedFeatures> = {};
      await Promise.all(
        vehicles.map(async (v) => {
          try {
            features[v.vehicle_id] = await fetchVehicleFeatures(v.vehicle_id);
          } catch {
            // vehicle may not be materialized yet
          }
        })
      );
      setVehicleFeatures(features);
    } catch {
      // silently ignore — vehicles tab may not be active
    }
  }, []);

  useEffect(() => {
    if (mounted) refreshVehicles();
  }, [mounted, refreshVehicles]);

  // Poll for pending vehicles until all are materialized
  useEffect(() => {
    const hasPending = savedVehicles.some(
      (v) => !vehicleFeatures[v.vehicle_id]?.materialized
    );
    if (!hasPending || savedVehicles.length === 0) return;

    const interval = setInterval(async () => {
      for (const v of savedVehicles) {
        if (vehicleFeatures[v.vehicle_id]?.materialized) continue;
        try {
          const feat = await fetchVehicleFeatures(v.vehicle_id);
          if (feat.materialized) {
            setVehicleFeatures((prev) => ({ ...prev, [v.vehicle_id]: feat }));
          }
        } catch {
          // ignore
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [savedVehicles, vehicleFeatures]);

  const priceRatio =
    vehicle.recommended_price > 0
      ? vehicle.actual_price / vehicle.recommended_price
      : 0;
  const priceDiff = vehicle.actual_price - vehicle.recommended_price;

  return (
    <main className="flex-1">
      {/* ── Hero header with Turo-inspired purple gradient ── */}
      <div className="bg-gradient-to-br from-[#593CFB] via-[#7C63FC] to-[#4429D4] text-white">
        <div className="mx-auto max-w-5xl px-6 py-12">
          <div className="flex items-start justify-between">
            <div>
              <div className="mb-2 flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/20 backdrop-blur-sm">
                  <Car className="h-5 w-5" />
                </div>
                <h1 className="text-3xl font-bold tracking-tight">
                  Vroom Forecast
                </h1>
              </div>
              <p className="max-w-md text-base text-white/75">
                Predict how many reservations a vehicle will get based on its
                listing attributes. Powered by ML.
              </p>
            </div>

            {/* Model status + reload */}
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger>
                  <div className="flex items-center gap-2 rounded-full bg-white/15 px-4 py-2 text-sm backdrop-blur-sm cursor-default">
                    <StatusDot online={!!health} />
                    {health ? (
                      <span>
                        Model v{health.model_version}
                      </span>
                    ) : (
                      <span className="text-white/60">Offline</span>
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {health
                    ? `Connected to ${health.model_name} on MLflow`
                    : "Cannot reach the prediction API"}
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={reloading || !health ? undefined : handleReload}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !reloading && health) handleReload();
                    }}
                    className={`flex h-9 w-9 items-center justify-center rounded-full bg-white/15 backdrop-blur-sm transition-colors ${reloading || !health ? "opacity-50 cursor-not-allowed" : "hover:bg-white/25 cursor-pointer"}`}
                  >
                    <RefreshCw className={`h-4 w-4 ${reloading ? "animate-spin" : ""}`} />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  Reload champion model from MLflow
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </div>
      </div>

      {/* ── Content area ── */}
      <div className="mx-auto max-w-5xl px-6 -mt-6">
        {error && (
          <div className="mb-4 rounded-xl border border-destructive/20 bg-destructive/5 p-4 text-sm text-destructive">
            {error}
          </div>
        )}

        <Tabs defaultValue="predict">
          <TabsList className="mb-6 rounded-full bg-white shadow-sm border">
            <TabsTrigger
              value="predict"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Predict
            </TabsTrigger>
            <TabsTrigger
              value="vehicles"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Fleet
            </TabsTrigger>
            <TabsTrigger
              value="benchmark"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Benchmark
            </TabsTrigger>
          </TabsList>

          {/* ── Predict Tab ── */}
          <TabsContent value="predict">
            <div className="grid gap-6 lg:grid-cols-5">
              {/* Vehicle form — takes 3/5 columns */}
              <Card className="lg:col-span-3 shadow-md border-0 bg-white">
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg">Vehicle Attributes</CardTitle>
                  <CardDescription>
                    Configure the vehicle listing to estimate reservations
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Pricing section */}
                  <div className="rounded-xl bg-muted/50 p-4 space-y-4">
                    <div className="flex items-center gap-2">
                      <DollarSign className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-semibold">Pricing</h3>
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <Label htmlFor="actual_price" className="text-xs">
                          Daily price
                        </Label>
                        <div className="relative">
                          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">
                            $
                          </span>
                          <Input
                            id="actual_price"
                            type="number"
                            min={1}
                            className="pl-7"
                            value={vehicle.actual_price}
                            onChange={(e) =>
                              update("actual_price", Number(e.target.value))
                            }
                          />
                        </div>
                      </div>
                      <div className="space-y-1.5">
                        <Label htmlFor="recommended_price" className="text-xs">
                          Recommended price
                        </Label>
                        <div className="relative">
                          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">
                            $
                          </span>
                          <Input
                            id="recommended_price"
                            type="number"
                            min={1}
                            className="pl-7"
                            value={vehicle.recommended_price}
                            onChange={(e) =>
                              update(
                                "recommended_price",
                                Number(e.target.value)
                              )
                            }
                          />
                        </div>
                      </div>
                    </div>
                    {vehicle.recommended_price > 0 && (
                      <div className="flex gap-3 text-xs">
                        <Badge
                          variant={
                            priceDiff <= 0 ? "secondary" : "destructive"
                          }
                          className="rounded-full font-mono"
                        >
                          {priceDiff >= 0 ? "+" : ""}${priceDiff.toFixed(0)} vs
                          recommended
                        </Badge>
                        <Badge variant="outline" className="rounded-full font-mono">
                          {priceRatio.toFixed(2)}x ratio
                        </Badge>
                      </div>
                    )}
                  </div>

                  {/* Listing quality section */}
                  <div className="space-y-5">
                    <div className="flex items-center gap-2">
                      <Image className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-semibold">Listing Quality</h3>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Photos</Label>
                        <span className="text-xs font-mono text-primary font-semibold">
                          {vehicle.num_images}
                        </span>
                      </div>
                      <Slider
                        min={0}
                        max={30}
                        step={1}
                        value={[vehicle.num_images]}
                        onValueChange={(v) =>
                          update("num_images", Array.isArray(v) ? v[0] : v)
                        }
                      />
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span>0</span>
                        <span>30</span>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Description length</Label>
                        <span className="text-xs font-mono text-primary font-semibold">
                          {vehicle.description} chars
                        </span>
                      </div>
                      <Slider
                        min={0}
                        max={1000}
                        step={10}
                        value={[vehicle.description]}
                        onValueChange={(v) =>
                          update("description", Array.isArray(v) ? v[0] : v)
                        }
                      />
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span>0</span>
                        <span>1000</span>
                      </div>
                    </div>
                  </div>

                  <Separator />

                  {/* Toggles */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Settings className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-semibold">Vehicle Details</h3>
                    </div>

                    <div className="flex items-center justify-between rounded-lg bg-muted/50 p-3">
                      <div>
                        <Label htmlFor="technology" className="text-sm">
                          Technology package
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          GPS, Bluetooth, etc.
                        </p>
                      </div>
                      <Switch
                        id="technology"
                        checked={vehicle.technology === 1}
                        onCheckedChange={(c) =>
                          update("technology", c ? 1 : 0)
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between rounded-lg bg-muted/50 p-3">
                      <div>
                        <Label htmlFor="street_parked" className="text-sm">
                          Street parked
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          No dedicated parking spot
                        </p>
                      </div>
                      <Switch
                        id="street_parked"
                        checked={vehicle.street_parked === 1}
                        onCheckedChange={(c) =>
                          update("street_parked", c ? 1 : 0)
                        }
                      />
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <Button
                      className="flex-1 rounded-full h-11 text-base font-semibold shadow-lg shadow-primary/25"
                      size="lg"
                      onClick={handlePredict}
                      disabled={loading || (mounted && !health)}
                    >
                      {loading ? (
                        <span className="flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          Predicting...
                        </span>
                      ) : (
                        "Predict"
                      )}
                    </Button>
                    <Tooltip>
                      <TooltipTrigger>
                        <div
                          role="button"
                          tabIndex={0}
                          onClick={saving || !health ? undefined : handleSaveVehicle}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !saving && health) handleSaveVehicle();
                          }}
                          className={`flex h-11 w-11 items-center justify-center rounded-full border ${saving || !health ? "opacity-50 cursor-not-allowed" : "hover:bg-muted cursor-pointer"}`}
                        >
                          <Save className={`h-5 w-5 ${saving ? "animate-pulse" : ""}`} />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>Save to fleet</TooltipContent>
                    </Tooltip>
                  </div>
                </CardContent>
              </Card>

              {/* Result — takes 2/5 columns */}
              <div className="lg:col-span-2 space-y-6">
                <Card className="shadow-md border-0 bg-white overflow-hidden">
                  <div className="bg-gradient-to-br from-primary/5 to-primary/10 p-8">
                    <CardDescription className="text-center mb-4">
                      Estimated Reservations
                    </CardDescription>
                    {prediction ? (
                      <div className="text-center">
                        <div className="text-7xl font-bold tabular-nums text-primary">
                          {prediction.predicted_reservations}
                        </div>
                        <p className="mt-1 text-sm text-muted-foreground">
                          reservations
                        </p>
                      </div>
                    ) : (
                      <div className="text-center py-4">
                        <div className="mx-auto mb-3 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                          <Monitor className="h-8 w-8 text-muted-foreground/50" />
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Configure your vehicle and click predict
                        </p>
                      </div>
                    )}
                  </div>
                  {prediction && (
                    <div className="border-t px-6 py-3 flex items-center justify-center">
                      <Badge
                        variant="outline"
                        className="rounded-full text-xs"
                      >
                        model v{prediction.model_version}
                      </Badge>
                    </div>
                  )}
                </Card>

                {/* Quick insights card */}
                <Card className="shadow-md border-0 bg-white">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Listing Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {[
                      {
                        label: "Price competitiveness",
                        value:
                          priceRatio <= 0.9
                            ? "Below market"
                            : priceRatio <= 1.1
                              ? "At market"
                              : "Above market",
                        color:
                          priceRatio <= 0.9
                            ? "text-turo-teal"
                            : priceRatio <= 1.1
                              ? "text-primary"
                              : "text-destructive",
                      },
                      {
                        label: "Photo quality",
                        value:
                          vehicle.num_images >= 15
                            ? "Excellent"
                            : vehicle.num_images >= 8
                              ? "Good"
                              : vehicle.num_images >= 3
                                ? "Fair"
                                : "Poor",
                        color:
                          vehicle.num_images >= 8
                            ? "text-turo-teal"
                            : "text-destructive",
                      },
                      {
                        label: "Description",
                        value:
                          vehicle.description >= 300
                            ? "Detailed"
                            : vehicle.description >= 100
                              ? "Basic"
                              : "Minimal",
                        color:
                          vehicle.description >= 200
                            ? "text-turo-teal"
                            : "text-destructive",
                      },
                    ].map(({ label, value, color }) => (
                      <div
                        key={label}
                        className="flex items-center justify-between text-sm"
                      >
                        <span className="text-muted-foreground">{label}</span>
                        <span className={`font-medium ${color}`}>{value}</span>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* ── Benchmark Tab ── */}
          <TabsContent value="benchmark">
            <div className="space-y-6">
              <Card className="shadow-md border-0 bg-white">
                <CardHeader>
                  <CardTitle className="text-lg">Latency Benchmark</CardTitle>
                  <CardDescription>
                    Compare inference latency between two paths: raw feature computation
                    vs online store lookup (Redis). Each runs 1,000 iterations measuring
                    end-to-end time from features to prediction.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-3">
                    <Button
                      onClick={handleBenchmark}
                      disabled={benchLoading || (mounted && !health)}
                      className="rounded-full shadow-lg shadow-primary/25"
                    >
                      {benchLoading ? "Running..." : "Raw Features"}
                    </Button>
                    <Button
                      onClick={handleBenchmarkStore}
                      disabled={benchStoreLoading || (mounted && !health)}
                      variant="outline"
                      className="rounded-full"
                    >
                      {benchStoreLoading ? "Running..." : "Online Store"}
                    </Button>
                  </div>
                  {(benchLoading || benchStoreLoading) && (
                    <Progress
                      value={benchLoading ? benchProgress : benchStoreProgress}
                      className="h-1.5 mt-4"
                    />
                  )}
                </CardContent>
              </Card>

              {/* Results side by side */}
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Raw features benchmark */}
                <Card className={`shadow-md border-0 bg-white ${benchmarkResult ? "" : "opacity-50"}`}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Badge className="rounded-full bg-primary text-primary-foreground">Raw</Badge>
                      <CardTitle className="text-sm">Feature Computation + Inference</CardTitle>
                    </div>
                    <CardDescription className="text-xs">
                      Features computed on the fly from request attributes
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {benchmarkResult ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          {[
                            { label: "avg", value: benchmarkResult.avg_latency_ms, highlight: true },
                            { label: "p50", value: benchmarkResult.p50_latency_ms, highlight: false },
                            { label: "p95", value: benchmarkResult.p95_latency_ms, highlight: false },
                            { label: "p99", value: benchmarkResult.p99_latency_ms, highlight: false },
                          ].map(({ label, value, highlight }) => (
                            <div key={label} className={`rounded-lg p-3 text-center ${highlight ? "bg-primary/5 border border-primary/20" : "bg-muted/50"}`}>
                              <div className={`text-xl font-bold tabular-nums ${highlight ? "text-primary" : ""}`}>
                                {value.toFixed(2)}
                              </div>
                              <p className="text-[10px] text-muted-foreground">{label} (ms)</p>
                            </div>
                          ))}
                        </div>
                        <p className="text-[10px] text-muted-foreground text-center">
                          {benchmarkResult.n_iterations.toLocaleString()} iterations &middot; model v{benchmarkResult.model_version}
                        </p>
                        <Separator className="my-3" />
                        <div className="grid grid-cols-2 gap-2 text-center">
                          <div className="rounded-lg bg-muted/30 p-2">
                            <p className="text-xs font-semibold tabular-nums">{benchmarkResult.avg_features_ms.toFixed(2)} ms</p>
                            <p className="text-[10px] text-muted-foreground">features (compute)</p>
                          </div>
                          <div className="rounded-lg bg-muted/30 p-2">
                            <p className="text-xs font-semibold tabular-nums">{benchmarkResult.avg_predict_ms.toFixed(2)} ms</p>
                            <p className="text-[10px] text-muted-foreground">predict</p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground text-center py-6">
                        Click &quot;Raw Features&quot; to run
                      </p>
                    )}
                  </CardContent>
                </Card>

                {/* Online store benchmark */}
                <Card className={`shadow-md border-0 bg-white ${benchmarkStoreResult ? "" : "opacity-50"}`}>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Badge className="rounded-full bg-turo-teal text-white">Store</Badge>
                      <CardTitle className="text-sm">Redis Lookup + Inference</CardTitle>
                    </div>
                    <CardDescription className="text-xs">
                      Pre-computed features read from the online store
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {benchmarkStoreResult ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                          {[
                            { label: "avg", value: benchmarkStoreResult.avg_latency_ms, highlight: true },
                            { label: "p50", value: benchmarkStoreResult.p50_latency_ms, highlight: false },
                            { label: "p95", value: benchmarkStoreResult.p95_latency_ms, highlight: false },
                            { label: "p99", value: benchmarkStoreResult.p99_latency_ms, highlight: false },
                          ].map(({ label, value, highlight }) => (
                            <div key={label} className={`rounded-lg p-3 text-center ${highlight ? "bg-turo-teal/10 border border-turo-teal/20" : "bg-muted/50"}`}>
                              <div className={`text-xl font-bold tabular-nums ${highlight ? "text-turo-teal" : ""}`}>
                                {value.toFixed(2)}
                              </div>
                              <p className="text-[10px] text-muted-foreground">{label} (ms)</p>
                            </div>
                          ))}
                        </div>
                        <p className="text-[10px] text-muted-foreground text-center">
                          {benchmarkStoreResult.n_iterations.toLocaleString()} iterations &middot; model v{benchmarkStoreResult.model_version}
                        </p>
                        <Separator className="my-3" />
                        <div className="grid grid-cols-2 gap-2 text-center">
                          <div className="rounded-lg bg-muted/30 p-2">
                            <p className="text-xs font-semibold tabular-nums">{benchmarkStoreResult.avg_features_ms.toFixed(2)} ms</p>
                            <p className="text-[10px] text-muted-foreground">features (Redis)</p>
                          </div>
                          <div className="rounded-lg bg-muted/30 p-2">
                            <p className="text-xs font-semibold tabular-nums">{benchmarkStoreResult.avg_predict_ms.toFixed(2)} ms</p>
                            <p className="text-[10px] text-muted-foreground">predict</p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground text-center py-6">
                        Click &quot;Online Store&quot; to run (requires a saved vehicle)
                      </p>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Comparison summary */}
              {benchmarkResult && benchmarkStoreResult && (
                <Card className="shadow-md border-0 bg-white">
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-center gap-6 text-sm">
                      <div className="text-center">
                        <p className="text-muted-foreground">Raw avg</p>
                        <p className="text-lg font-bold tabular-nums text-primary">
                          {benchmarkResult.avg_latency_ms.toFixed(2)} ms
                        </p>
                      </div>
                      <span className="text-muted-foreground">vs</span>
                      <div className="text-center">
                        <p className="text-muted-foreground">Store avg</p>
                        <p className="text-lg font-bold tabular-nums text-turo-teal">
                          {benchmarkStoreResult.avg_latency_ms.toFixed(2)} ms
                        </p>
                      </div>
                      <Separator orientation="vertical" className="h-10" />
                      <div className="text-center">
                        <p className="text-muted-foreground">Difference</p>
                        <p className={`text-lg font-bold tabular-nums ${benchmarkStoreResult.avg_latency_ms < benchmarkResult.avg_latency_ms ? "text-turo-teal" : "text-destructive"}`}>
                          {benchmarkStoreResult.avg_latency_ms < benchmarkResult.avg_latency_ms ? "" : "+"}
                          {(benchmarkStoreResult.avg_latency_ms - benchmarkResult.avg_latency_ms).toFixed(2)} ms
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* ── Vehicles Tab (Fleet Dashboard) ── */}
          <TabsContent value="vehicles">
            <Card className="shadow-md border-0 bg-white">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-lg">Fleet</CardTitle>
                    <CardDescription>
                      Saved vehicles with predictions from the online feature store.
                      Save vehicles from the Predict tab using the
                      <Save className="inline h-3.5 w-3.5 mx-1 -mt-0.5" />
                      button.
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2">
                    {savedVehicles.length > 0 && (
                      <Button
                        size="sm"
                        className="rounded-full"
                        onClick={handlePredictAll}
                        disabled={mounted && !health}
                      >
                        Predict All
                      </Button>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={refreshVehicles}
                      className="rounded-full"
                    >
                      Refresh
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {savedVehicles.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="mx-auto mb-3 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                      <Car className="h-8 w-8 text-muted-foreground/50" />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      No vehicles in your fleet yet.
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Configure a vehicle in the Predict tab and save it.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {savedVehicles.map((v) => {
                      const pred = vehiclePredictions[v.vehicle_id];
                      const feat = vehicleFeatures[v.vehicle_id];
                      return (
                        <div
                          key={v.vehicle_id}
                          className="rounded-xl border overflow-hidden hover:bg-muted/30 transition-colors"
                        >
                          <div className="flex items-center justify-between p-4">
                            <div className="space-y-1.5">
                              <div className="flex items-center gap-2">
                                <span className="font-semibold">
                                  #{v.vehicle_id}
                                </span>
                                <Badge variant="outline" className="rounded-full text-xs font-mono">
                                  ${v.actual_price}/day
                                </Badge>
                                {v.technology === 1 && (
                                  <Badge variant="secondary" className="rounded-full text-xs">
                                    Tech
                                  </Badge>
                                )}
                                {v.street_parked === 1 && (
                                  <Badge variant="secondary" className="rounded-full text-xs">
                                    Street
                                  </Badge>
                                )}
                                {feat?.materialized ? (
                                  <Badge className="rounded-full text-xs bg-turo-teal text-white">
                                    Materialized
                                  </Badge>
                                ) : (
                                  <Badge variant="outline" className="rounded-full text-xs text-muted-foreground">
                                    Pending
                                  </Badge>
                                )}
                              </div>
                              <p className="text-xs text-muted-foreground">
                                Rec. ${v.recommended_price} &middot;{" "}
                                {v.num_images} photos &middot;{" "}
                                {v.description} chars description
                              </p>
                            </div>
                            <div className="flex items-center gap-4">
                              {pred ? (
                                <div className="text-right">
                                  <div className="text-2xl font-bold tabular-nums text-primary">
                                    {pred.predicted_reservations}
                                  </div>
                                  <p className="text-[10px] text-muted-foreground">
                                    reservations (v{pred.model_version})
                                  </p>
                                </div>
                              ) : (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="rounded-full"
                                  onClick={() => handlePredictById(v.vehicle_id)}
                                  disabled={(mounted && !health) || !feat?.materialized}
                                >
                                  Predict
                                </Button>
                              )}
                            </div>
                          </div>

                          {/* Computed features panel */}
                          {feat?.materialized && (
                            <div className="border-t bg-muted/20 px-4 py-3">
                              <p className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                                Computed Features (from online store)
                              </p>
                              <div className="grid grid-cols-4 gap-3">
                                <div>
                                  <p className="text-[10px] text-muted-foreground">price_diff</p>
                                  <p className="text-sm font-mono font-semibold">
                                    {feat.price_diff !== null ? (feat.price_diff >= 0 ? "+" : "") + feat.price_diff.toFixed(2) : "—"}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-[10px] text-muted-foreground">price_ratio</p>
                                  <p className="text-sm font-mono font-semibold">
                                    {feat.price_ratio !== null ? feat.price_ratio.toFixed(3) : "—"}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-[10px] text-muted-foreground">technology</p>
                                  <p className="text-sm font-mono font-semibold">
                                    {feat.technology ?? "—"}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-[10px] text-muted-foreground">num_images</p>
                                  <p className="text-sm font-mono font-semibold">
                                    {feat.num_images ?? "—"}
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* ── Footer ── */}
      <footer className="mt-16 border-t">
        <div className="mx-auto max-w-5xl px-6 py-6 flex items-center justify-between text-xs text-muted-foreground">
          <span>Vroom Forecast &mdash; MLOps Take-Home</span>
          <div className="flex items-center gap-4">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors"
            >
              API Docs
            </a>
            <a
              href="http://localhost:5001"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors"
            >
              MLflow
            </a>
            <a
              href="http://localhost:8080"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors"
            >
              Airflow
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
