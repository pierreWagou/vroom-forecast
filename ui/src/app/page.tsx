"use client";

import { useState, useEffect, useCallback } from "react";
import { useTheme } from "next-themes";
import {
  BookOpen,
  Car,
  DollarSign,
  HardDrive,
  ImageIcon,
  Loader2,
  Monitor,
  Moon,
  Plus,
  RefreshCw,
  Save,
  Server,
  Settings,
  Sun,
  Trash2,
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
  deleteVehicle,
  listVehicles,
  fetchVehicleFeatures,
  fetchAllVehicleFeatures,
  fetchStoreInfo,
  API_URL,
  type VehicleFeatures,
  type PredictionResponse,
  type HealthResponse,
  type BenchmarkResponse,
  type VehicleRecord,
  type ComputedFeatures,
  type StoreInfo,
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
  const { setTheme, resolvedTheme } = useTheme();
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
  const [benchVehicleId, setBenchVehicleId] = useState<string>("");
  const [benchProgress, setBenchProgress] = useState(0);
  const [benchStoreProgress, setBenchStoreProgress] = useState(0);
  const [benchIterations, setBenchIterations] = useState("1000");
  const [activeTab, setActiveTab] = useState("predict");
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedVehicles, setSavedVehicles] = useState<VehicleRecord[]>([]);
  const [fleetSearch, setFleetSearch] = useState("");
  const [vehiclePredictions, setVehiclePredictions] = useState<
    Record<number, PredictionResponse>
  >({});
  const [vehicleFeatures, setVehicleFeatures] = useState<
    Record<number, ComputedFeatures>
  >({});
  const [storeInfo, setStoreInfo] = useState<StoreInfo | null>(null);

  useEffect(() => {
    setMounted(true);
    fetchHealth().then(setHealth).catch(() => setHealth(null));
    fetchStoreInfo().then(setStoreInfo).catch(() => {});
  }, []);

  // Subscribe to SSE for model-promoted events — auto-refreshes health when
  // a new champion is promoted, without polling. Also handles the initial
  // "offline → online" transition by retrying the EventSource connection.
  useEffect(() => {
    if (!mounted) return;

    const apiUrl = API_URL;
    let es: EventSource | null = null;
    let retryTimeout: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      es = new EventSource(`${apiUrl}/events`);
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "model-promoted") {
            // Refresh health to get the new model version
            fetchHealth().then(setHealth).catch(() => {});
          }
        } catch {
          // Ignore parse errors (keepalive comments etc.)
        }
      };
      es.onerror = () => {
        // Connection lost — close and retry in 5s
        es?.close();
        retryTimeout = setTimeout(connect, 5000);
      };
    };

    connect();
    return () => {
      es?.close();
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, [mounted]);

  const handleReload = async () => {
    setReloading(true);
    setError(null);
    try {
      await reloadModel();
      // Refresh health to get updated version
      const h = await fetchHealth();
      setHealth(h);
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
      setError(e instanceof Error ? e.message : "Simulation failed");
    } finally {
      setLoading(false);
    }
  }, [vehicle]);

  const handleBenchmark = async () => {
    setBenchLoading(true);
    setBenchProgress(0);
    setError(null);
    const iterations = Math.max(10, Number(benchIterations) || 1000);
    const estimatedMs = iterations * 0.015 * 1000; // ~15ms per iteration
    const start = Date.now();
    const timer = setInterval(() => {
      const elapsed = Date.now() - start;
      setBenchProgress(Math.min(Math.round((elapsed / estimatedMs) * 95), 95));
    }, 100);
    try {
      const result = await benchmark(vehicle, iterations);
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
    const vid = Number(benchVehicleId);
    if (!vid || vid <= 0) {
      setError("Enter a valid vehicle ID to benchmark.");
      return;
    }
    setBenchStoreLoading(true);
    setBenchStoreProgress(0);
    setError(null);
    const iterations = Math.max(10, Number(benchIterations) || 1000);
    const estimatedMs = iterations * 0.015 * 1000;
    const start = Date.now();
    const timer = setInterval(() => {
      const elapsed = Date.now() - start;
      setBenchStoreProgress(Math.min(Math.round((elapsed / estimatedMs) * 95), 95));
    }, 100);
    try {
      const result = await benchmarkById(vid, iterations);
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
      await refreshVehicles();
      setActiveTab("vehicles");
      if (!result.event_published) {
        setError(
          `Vehicle #${result.vehicle_id} saved, but feature materialization could not be triggered (Redis unavailable).`
        );
      }
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
      if (e instanceof Error) {
        if (e.message.startsWith("[503]")) {
          setError("No model loaded. Train and promote a model first.");
        } else if (e.message.startsWith("[404]")) {
          setError(
            `Vehicle #${vehicleId} not yet materialized. Run the feature pipeline first.`
          );
        } else {
          setError(e.message);
        }
      } else {
        setError("Simulation failed");
      }
    }
  };

  const handlePredictAll = async () => {
    setError(null);
    const arrivals = savedVehicles.filter((v) => v.num_reservations === null);
    const results = await Promise.allSettled(
      arrivals.map((v) => predictById(v.vehicle_id))
    );
    const newPredictions: Record<number, PredictionResponse> = {};
    results.forEach((result, i) => {
      if (result.status === "fulfilled") {
        newPredictions[arrivals[i].vehicle_id] = result.value;
      }
      // Skip vehicles that failed (not yet materialized)
    });
    setVehiclePredictions((prev) => ({ ...prev, ...newPredictions }));
  };

  const handleDeleteVehicle = async (vehicleId: number) => {
    setError(null);
    try {
      await deleteVehicle(vehicleId);
      await refreshVehicles();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const refreshVehicles = useCallback(async () => {
    try {
      const vehicles = await listVehicles();
      setSavedVehicles(vehicles);
      // Fetch computed features for all vehicles in a single batch call
      const allFeatures = await fetchAllVehicleFeatures();
      const features: Record<number, ComputedFeatures> = {};
      for (const f of allFeatures) {
        features[f.vehicle_id] = f;
      }
      setVehicleFeatures(features);
    } catch {
      // silently ignore — vehicles tab may not be active
    }
  }, []);

  useEffect(() => {
    if (mounted) refreshVehicles();
  }, [mounted, refreshVehicles]);

  // SSE: persistent connection for vehicle-materialized events.
  // Stays open as long as the component is mounted — doesn't reconnect on
  // state changes, so we never miss events due to race conditions.
  useEffect(() => {
    if (!mounted) return;

    const es = new EventSource(`${API_URL}/vehicles/events`);

    es.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "vehicle-materialized" && data.vehicle_id) {
          const feat = await fetchVehicleFeatures(data.vehicle_id);
          if (feat.materialized) {
            setVehicleFeatures((prev) => ({ ...prev, [data.vehicle_id]: feat }));
          }
        }
      } catch {
        // Ignore parse errors (keepalive comments etc.)
      }
    };

    return () => es.close();
  }, [mounted]);

  // Auto-pick first materialized vehicle for the online store benchmark
  useEffect(() => {
    if (benchVehicleId) return; // already set
    const materialized = savedVehicles.find(
      (v) => vehicleFeatures[v.vehicle_id]?.materialized
    );
    if (materialized) {
      setBenchVehicleId(String(materialized.vehicle_id));
    }
  }, [savedVehicles, vehicleFeatures, benchVehicleId]);

  const priceRatio =
    vehicle.recommended_price > 0
      ? vehicle.actual_price / vehicle.recommended_price
      : 0;
  const priceDiff = vehicle.actual_price - vehicle.recommended_price;

  const filteredVehicles = savedVehicles.filter((v) => {
    if (!fleetSearch) return true;
    const q = fleetSearch.toLowerCase();
    return (
      v.vehicle_id.toString().includes(q) ||
      v.actual_price.toString().includes(q)
    );
  });
  const fleetVehicles = filteredVehicles.filter((v) => v.num_reservations !== null);
  const newArrivals = filteredVehicles.filter((v) => v.num_reservations === null);
  const displayedFleet = fleetVehicles.slice(0, 50);
  const displayedArrivals = newArrivals.slice(0, 50);

  return (
    <main className="flex-1">
      {/* ── Hero header with Turo-inspired purple gradient ── */}
      <div className="bg-gradient-to-br from-[#593CFB] via-[#7C63FC] to-[#4429D4] dark:from-[#3D2AB0] dark:via-[#593CFB] dark:to-[#2E1C8A] text-white">
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
                Forecast vehicle booking demand from listing attributes — pricing,
                photos, description, and more.
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

              <Tooltip>
                <TooltipTrigger>
                  <a
                    href="http://localhost:8100"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex h-9 w-9 items-center justify-center rounded-full bg-white/15 backdrop-blur-sm transition-colors hover:bg-white/25"
                  >
                    <BookOpen className="h-4 w-4" />
                  </a>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  Documentation
                </TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") setTheme(resolvedTheme === "dark" ? "light" : "dark");
                    }}
                    className="flex h-9 w-9 items-center justify-center rounded-full bg-white/15 backdrop-blur-sm transition-colors hover:bg-white/25 cursor-pointer"
                  >
                    {mounted && resolvedTheme === "dark" ? (
                      <Sun className="h-4 w-4" />
                    ) : (
                      <Moon className="h-4 w-4" />
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom">
                  {mounted && resolvedTheme === "dark" ? "Light mode" : "Dark mode"}
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

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6 rounded-full bg-card shadow-sm border">
            <TabsTrigger
              value="predict"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Simulation
            </TabsTrigger>
            <TabsTrigger
              value="vehicles"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Catalog
            </TabsTrigger>
            <TabsTrigger
              value="benchmark"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Benchmark
            </TabsTrigger>
            <TabsTrigger
              value="features"
              className="rounded-full data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              Feature Store
            </TabsTrigger>
          </TabsList>

          {/* ── Simulation Tab ── */}
          <TabsContent value="predict">
            <div className="grid gap-6 lg:grid-cols-5">
              {/* Vehicle form — takes 3/5 columns */}
              <Card className="lg:col-span-3 shadow-md border-0 bg-card">
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
                      <ImageIcon className="h-4 w-4 text-primary" />
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
                          Simulating...
                        </span>
                      ) : (
                        "Simulate"
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
                      <TooltipContent>Save to catalog</TooltipContent>
                    </Tooltip>
                  </div>
                </CardContent>
              </Card>

              {/* Result — takes 2/5 columns */}
              <div className="lg:col-span-2 space-y-6">
                <Card className="shadow-md border-0 bg-card overflow-hidden">
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
                <Card className="shadow-md border-0 bg-card">
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
              <Card className="shadow-md border-0 bg-card">
                <CardHeader>
                  <CardTitle className="text-lg">Latency Benchmark</CardTitle>
                  <CardDescription>
                    Compare inference latency between two paths: raw feature computation
                    vs online store lookup (Redis).
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex gap-3 items-end">
                    <div className="space-y-1">
                      <Label htmlFor="bench-iterations" className="text-xs">Iterations</Label>
                      <Input
                        id="bench-iterations"
                        type="number"
                        min={10}
                        max={100000}
                        className="w-28 h-9"
                        value={benchIterations}
                        onChange={(e) => setBenchIterations(e.target.value)}
                        onBlur={() => {
                          const n = Number(benchIterations);
                          if (!n || n < 10) setBenchIterations("10");
                          else if (n > 100000) setBenchIterations("100000");
                          else setBenchIterations(String(Math.round(n)));
                        }}
                        disabled={benchLoading && benchStoreLoading}
                      />
                    </div>
                    <Button
                      onClick={handleBenchmark}
                      disabled={benchLoading || (mounted && !health)}
                      className="rounded-full shadow-lg shadow-primary/25"
                    >
                      {benchLoading ? "Running..." : "Raw Features"}
                    </Button>
                    <Button
                      onClick={handleBenchmarkStore}
                      disabled={benchStoreLoading || (mounted && !health) || !benchVehicleId}
                      className="rounded-full bg-turo-teal text-white shadow-lg shadow-turo-teal/25 hover:bg-turo-teal/90"
                    >
                      {benchStoreLoading ? "Running..." : "Online Store"}
                    </Button>
                    {!benchVehicleId && savedVehicles.length > 0 && (
                      <p className="text-xs text-muted-foreground">
                        No materialized vehicle found — run the feature pipeline first.
                      </p>
                    )}
                    {savedVehicles.length === 0 && (
                      <p className="text-xs text-muted-foreground">
                        Save a vehicle in the Catalog to enable Online Store benchmark.
                      </p>
                    )}
                  </div>
                  {(benchLoading || benchStoreLoading) && (
                    <div className="mt-4 space-y-2">
                      {benchLoading && (
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-muted-foreground w-16 shrink-0">Raw</span>
                          <Progress
                            value={benchProgress}
                            className="h-1.5 flex-1 [&_[data-slot=progress-indicator]]:bg-primary"
                          />
                        </div>
                      )}
                      {benchStoreLoading && (
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] text-turo-teal w-16 shrink-0">Online</span>
                          <Progress
                            value={benchStoreProgress}
                            className="h-1.5 flex-1 [&_[data-slot=progress-indicator]]:bg-turo-teal"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Results side by side */}
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Raw features benchmark */}
                <Card className={`shadow-md border-0 bg-card ${benchmarkResult ? "" : "opacity-50"}`}>
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
                <Card className={`shadow-md border-0 bg-card ${benchmarkStoreResult ? "" : "opacity-50"}`}>
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
                        Click &quot;Online Store&quot; to run
                      </p>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Comparison summary */}
              {benchmarkResult && benchmarkStoreResult && (
                <Card className="shadow-md border-0 bg-card">
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

          {/* ── Catalog Tab ── */}
          <TabsContent value="vehicles">
            <div className="space-y-6">
              {/* Header card with search and actions */}
              <Card className="shadow-md border-0 bg-card">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">
                        Vehicle Catalog
                        <span className="ml-2 text-sm font-normal text-muted-foreground">
                          {savedVehicles.length} vehicles
                        </span>
                      </CardTitle>
                      <CardDescription>
                        Browse fleet vehicles with reservation history and
                        new arrivals awaiting their first bookings.
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                {savedVehicles.length > 0 && (
                  <CardContent className="pt-0">
                    <Input
                      placeholder="Search by ID or price..."
                      value={fleetSearch}
                      onChange={(e) => setFleetSearch(e.target.value)}
                      className="max-w-xs"
                    />
                  </CardContent>
                )}
              </Card>

              {savedVehicles.length === 0 ? (
                <Card className="shadow-md border-0 bg-card">
                  <CardContent className="pt-6">
                    <div className="text-center py-12">
                      <div className="mx-auto mb-3 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                        <Car className="h-8 w-8 text-muted-foreground/50" />
                      </div>
                      <p className="text-sm text-muted-foreground">
                        No vehicles in your catalog yet.
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Configure a vehicle in the Simulation tab and save it, or
                        run the feature pipeline to seed the catalog.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <>
                  {/* ── New Arrivals Section ── */}
                  {newArrivals.length > 0 && (
                    <Card className="shadow-md border-0 bg-card">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-turo-teal/10">
                              <Plus className="h-4 w-4 text-turo-teal" />
                            </div>
                            <CardTitle className="text-base">
                              New Arrivals
                              <span className="ml-2 text-sm font-normal text-muted-foreground">
                                {newArrivals.length} vehicles
                              </span>
                            </CardTitle>
                          </div>
                          <Button
                            size="sm"
                            className="rounded-full"
                            onClick={handlePredictAll}
                            disabled={mounted && !health}
                          >
                            Simulate All
                          </Button>
                        </div>
                        <CardDescription className="text-xs">
                          Vehicles with no reservation history yet &mdash; features available after real-time materialization or next pipeline run
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        {displayedArrivals.length < newArrivals.length && (
                          <p className="text-xs text-muted-foreground mb-3">
                            Showing 50 of {newArrivals.length} vehicles. Use search to narrow down.
                          </p>
                        )}
                        <div className="space-y-3">
                          {displayedArrivals.map((v) => {
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
                                      {feat?.materialized ? (
                                        <Badge variant="outline" className={`rounded-full text-xs ${feat.store === "online" ? "text-turo-teal border-turo-teal/30" : "text-primary border-primary/30"}`}>
                                          {feat.store === "online" ? "Online Store" : "Offline Store"}
                                        </Badge>
                                      ) : (
                                        <Badge variant="outline" className="rounded-full text-xs text-amber-500 border-amber-500/30">
                                          Pending
                                        </Badge>
                                      )}
                                    </div>
                                    <p className="text-xs text-muted-foreground">
                                      Rec. ${v.recommended_price} &middot;{" "}
                                      {v.num_images} photos &middot;{" "}
                                      {v.description} chars
                                    </p>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    {pred ? (
                                      <div className="text-right">
                                        <div className="text-2xl font-bold tabular-nums text-primary">
                                          {pred.predicted_reservations}
                                        </div>
                                        <p className="text-[10px] text-muted-foreground">
                                          predicted (v{pred.model_version})
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
                                        Simulate
                                      </Button>
                                    )}
                                    <button
                                      type="button"
                                      title="Remove vehicle"
                                      className="h-8 w-8 p-0 flex items-center justify-center rounded-md text-muted-foreground hover:text-destructive hover:bg-accent transition-colors cursor-pointer"
                                      onClick={() => handleDeleteVehicle(v.vehicle_id)}
                                    >
                                      <Trash2 className="h-3.5 w-3.5" />
                                    </button>
                                  </div>
                                </div>

                                {/* Computed features panel */}
                                {feat?.materialized && (
                                  <div className="border-t bg-muted/20 px-4 py-3">
                                    <p className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                                      Features
                                      <span className="ml-1.5 font-normal normal-case">
                                        {feat.store === "online"
                                          ? "computed on save, read from Redis"
                                          : feat.store === "offline"
                                            ? "loaded from offline store (Parquet)"
                                            : ""}
                                      </span>
                                    </p>
                                    <div className="grid grid-cols-5 gap-3">
                                      <div>
                                        <p className="text-[10px] text-muted-foreground">technology</p>
                                        <p className="text-sm font-mono font-semibold">
                                          {feat.technology ?? "\u2014"}
                                        </p>
                                      </div>
                                      <div>
                                        <p className="text-[10px] text-muted-foreground">num_images</p>
                                        <p className="text-sm font-mono font-semibold">
                                          {feat.num_images ?? "\u2014"}
                                        </p>
                                      </div>
                                      <div>
                                        <p className="text-[10px] text-muted-foreground">street_parked</p>
                                        <p className="text-sm font-mono font-semibold">
                                          {feat.street_parked ?? "\u2014"}
                                        </p>
                                      </div>
                                      <div>
                                        <p className="text-[10px] text-muted-foreground">description</p>
                                        <p className="text-sm font-mono font-semibold">
                                          {feat.description ?? "\u2014"}
                                        </p>
                                      </div>
                                      <div>
                                        <p className="text-[10px] text-muted-foreground">price_diff</p>
                                        <p className="text-sm font-mono font-semibold">
                                          {feat.price_diff !== null ? (feat.price_diff >= 0 ? "+" : "") + feat.price_diff.toFixed(2) : "\u2014"}
                                        </p>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* ── Fleet Section ── */}
                  <Card className="shadow-md border-0 bg-card">
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
                          <Car className="h-4 w-4 text-primary" />
                        </div>
                        <CardTitle className="text-base">
                          Fleet
                          <span className="ml-2 text-sm font-normal text-muted-foreground">
                            {fleetVehicles.length} vehicles
                          </span>
                        </CardTitle>
                      </div>
                      <CardDescription className="text-xs">
                        Vehicles with observed reservation history &mdash; features from the batch pipeline
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {fleetVehicles.length === 0 ? (
                        <p className="text-sm text-muted-foreground text-center py-6">
                          {fleetSearch
                            ? "No fleet vehicles match your search."
                            : "No fleet vehicles yet. Run the feature pipeline to seed the dataset."}
                        </p>
                      ) : (
                        <>
                          {displayedFleet.length < fleetVehicles.length && (
                            <p className="text-xs text-muted-foreground mb-3">
                              Showing 50 of {fleetVehicles.length} vehicles. Use search to narrow down.
                            </p>
                          )}
                          <div className="space-y-3">
                            {displayedFleet.map((v) => {
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
                                        {feat?.materialized ? (
                                          <Badge variant="outline" className={`rounded-full text-xs ${feat.store === "online" ? "text-turo-teal border-turo-teal/30" : "text-primary border-primary/30"}`}>
                                            {feat.store === "offline" ? "Offline Store" : "Online Store"}
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
                                        {v.description} chars
                                      </p>
                                    </div>
                                    <div className="flex items-center gap-4">
                                      <div className="text-right">
                                        <div className="text-2xl font-bold tabular-nums text-primary">
                                          {v.num_reservations}
                                        </div>
                                        <p className="text-[10px] text-muted-foreground">
                                          reservations
                                        </p>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Computed features panel */}
                                  {feat?.materialized && (
                                    <div className="border-t bg-muted/20 px-4 py-3">
                                      <p className="text-[10px] font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                                        Features
                                        <span className="ml-1.5 font-normal normal-case">
                                          {feat.store === "online"
                                            ? "computed on save, read from Redis"
                                            : feat.store === "offline"
                                              ? "loaded from offline store (Parquet)"
                                              : ""}
                                        </span>
                                      </p>
                                      <div className="grid grid-cols-5 gap-3">
                                        <div>
                                          <p className="text-[10px] text-muted-foreground">technology</p>
                                          <p className="text-sm font-mono font-semibold">
                                            {feat.technology ?? "\u2014"}
                                          </p>
                                        </div>
                                        <div>
                                          <p className="text-[10px] text-muted-foreground">num_images</p>
                                          <p className="text-sm font-mono font-semibold">
                                            {feat.num_images ?? "\u2014"}
                                          </p>
                                        </div>
                                        <div>
                                          <p className="text-[10px] text-muted-foreground">street_parked</p>
                                          <p className="text-sm font-mono font-semibold">
                                            {feat.street_parked ?? "\u2014"}
                                          </p>
                                        </div>
                                        <div>
                                          <p className="text-[10px] text-muted-foreground">description</p>
                                          <p className="text-sm font-mono font-semibold">
                                            {feat.description ?? "\u2014"}
                                          </p>
                                        </div>
                                        <div>
                                          <p className="text-[10px] text-muted-foreground">price_diff</p>
                                          <p className="text-sm font-mono font-semibold">
                                            {feat.price_diff !== null ? (feat.price_diff >= 0 ? "+" : "") + feat.price_diff.toFixed(2) : "\u2014"}
                                          </p>
                                        </div>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </>
                      )}
                    </CardContent>
                  </Card>
                </>
              )}
            </div>
          </TabsContent>

          {/* ── Feature Store Tab ── */}
          <TabsContent value="features">
            {(() => {
              const allFeatures = Object.values(vehicleFeatures);
              const offlineCount = allFeatures.filter((f) => f.store === "offline").length;
              const onlineCount = allFeatures.filter((f) => f.store === "online").length;
              const pendingCount = savedVehicles.length - allFeatures.filter((f) => f.materialized).length;

              return (
                <div className="space-y-6">
                  {/* Store overview cards */}
                  <div className="grid gap-6 md:grid-cols-2">
                    {/* Offline store */}
                    <Card className="shadow-md border-0 bg-card">
                      <CardHeader className="pb-3">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
                            <HardDrive className="h-5 w-5 text-primary" />
                          </div>
                          <div>
                            <CardTitle className="text-base">Offline Store</CardTitle>
                            <CardDescription className="text-xs">Batch features for training &amp; fleet display</CardDescription>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Status</span>
                            <Badge variant="outline" className={`rounded-full text-xs ${storeInfo?.offline_store.available ? "text-primary border-primary/30" : "text-destructive border-destructive/30"}`}>
                              {storeInfo?.offline_store.available ? "Available" : "Unavailable"}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Type</span>
                            <span className="text-sm font-medium font-mono">{storeInfo?.offline_store.type ?? "file (Parquet)"}</span>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Vehicles</span>
                            <span className="text-sm font-bold tabular-nums text-primary">{offlineCount}</span>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Location</span>
                            <span className="text-xs font-mono text-muted-foreground truncate max-w-[200px]">{storeInfo?.offline_store.path ?? "\u2014"}</span>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Size</span>
                            <span className="text-sm font-mono">{storeInfo?.offline_store.size_bytes != null ? (storeInfo.offline_store.size_bytes / 1024).toFixed(1) + " KB" : "\u2014"}</span>
                          </div>
                          <div className="flex items-center justify-between py-2">
                            <span className="text-sm text-muted-foreground">Last modified</span>
                            <span className="text-xs font-mono text-muted-foreground">{storeInfo?.offline_store.last_modified != null ? new Date(storeInfo.offline_store.last_modified * 1000).toLocaleString() : "\u2014"}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Online store */}
                    <Card className="shadow-md border-0 bg-card">
                      <CardHeader className="pb-3">
                        <div className="flex items-center gap-3">
                          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-turo-teal/10">
                            <Server className="h-5 w-5 text-turo-teal" />
                          </div>
                          <div>
                            <CardTitle className="text-base">Online Store</CardTitle>
                            <CardDescription className="text-xs">Real-time features for inference on new arrivals</CardDescription>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Status</span>
                            <Badge variant="outline" className={`rounded-full text-xs ${storeInfo?.online_store.available ? "text-turo-teal border-turo-teal/30" : "text-destructive border-destructive/30"}`}>
                              {storeInfo?.online_store.available ? "Connected" : "Disconnected"}
                            </Badge>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Type</span>
                            <span className="text-sm font-medium font-mono">{storeInfo?.online_store.type ?? "redis"}</span>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Vehicles</span>
                            <span className="text-sm font-bold tabular-nums text-turo-teal">{onlineCount}</span>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Location</span>
                            <span className="text-xs font-mono text-muted-foreground truncate max-w-[200px]">{storeInfo?.online_store.redis_url ?? "\u2014"}</span>
                          </div>
                          <div className="flex items-center justify-between py-2 border-b">
                            <span className="text-sm text-muted-foreground">Size</span>
                            <span className="text-sm font-mono">{storeInfo?.online_store.used_memory_human ?? "\u2014"}</span>
                          </div>
                          <div className="flex items-center justify-between py-2">
                            <span className="text-sm text-muted-foreground">Pending</span>
                            <span className={`text-sm font-bold tabular-nums ${pendingCount > 0 ? "text-amber-500" : ""}`}>{pendingCount}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Feature view definition */}
                  {storeInfo?.feature_view && (
                    <Card className="shadow-md border-0 bg-card">
                      <CardHeader>
                        <CardTitle className="text-base">Feature View: {storeInfo.feature_view.name}</CardTitle>
                        <CardDescription className="text-xs">
                          Entity: <span className="font-mono">{storeInfo.feature_view.entity}</span> (key: <span className="font-mono">{storeInfo.feature_view.entity_key}</span>) &middot; TTL: {storeInfo.feature_view.ttl_days} days
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                          {storeInfo.feature_view.features.map((feat) => {
                            const isDerived = feat === "price_diff";
                            return (
                              <div
                                key={feat}
                                className={`rounded-lg border px-3 py-2 ${
                                  isDerived
                                    ? "border-turo-teal/30 bg-turo-teal/5"
                                    : "bg-muted/30"
                                }`}
                              >
                                <p className={`text-xs font-mono font-medium ${isDerived ? "text-turo-teal" : ""}`}>{feat}</p>
                                {isDerived ? (
                                  <p className="text-[10px] text-muted-foreground font-mono">
                                    actual_price − recommended_price
                                  </p>
                                ) : (
                                  <p className="text-[10px] text-muted-foreground">raw attribute</p>
                                )}
                              </div>
                            );
                          })}
                          <div className="rounded-lg border border-dashed bg-muted/10 px-3 py-2">
                            <p className="text-xs font-mono font-medium text-muted-foreground">{storeInfo.feature_view.label}</p>
                            <p className="text-[10px] text-muted-foreground">label (target)</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              );
            })()}
          </TabsContent>
        </Tabs>
      </div>

      {/* ── Footer ── */}
      <footer className="mt-16 border-t">
        <div className="mx-auto max-w-5xl px-6 py-6 flex items-center justify-between text-xs text-muted-foreground">
          <span>Vroom Forecast &mdash; MLOps Take-Home</span>
          <div className="flex items-center gap-4">
            <a
              href="http://localhost:8100"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-primary transition-colors"
            >
              Docs
            </a>
            <a
              href={`${API_URL}/docs`}
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
