"use client";

import { useState, useEffect, useCallback } from "react";
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
  fetchHealth,
  benchmark,
  type VehicleFeatures,
  type PredictionResponse,
  type HealthResponse,
  type BenchmarkResponse,
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
  const [loading, setLoading] = useState(false);
  const [benchLoading, setBenchLoading] = useState(false);
  const [benchProgress, setBenchProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

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

  const update = (field: keyof VehicleFeatures, value: number) =>
    setVehicle((prev) => ({ ...prev, [field]: value }));

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
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-5 w-5"
                  >
                    <path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-2-2.2-3.3C12.4 5 10.5 4 8.8 4H6.4c-1.1 0-2.1.6-2.7 1.4L2 8H1v2h1l.7 1H1v2h1v3c0 .6.4 1 1 1h2" />
                    <circle cx="7" cy="17" r="2" />
                    <circle cx="17" cy="17" r="2" />
                  </svg>
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

            {/* Model status pill */}
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
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="h-4 w-4 text-primary"
                      >
                        <line x1="12" y1="1" x2="12" y2="23" />
                        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                      </svg>
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
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="h-4 w-4 text-primary"
                      >
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                        <circle cx="9" cy="9" r="2" />
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                      </svg>
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
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="h-4 w-4 text-primary"
                      >
                        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
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

                  <Button
                    className="w-full rounded-full h-11 text-base font-semibold shadow-lg shadow-primary/25"
                    size="lg"
                    onClick={handlePredict}
                    disabled={loading || !health}
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                            fill="none"
                          />
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                          />
                        </svg>
                        Predicting...
                      </span>
                    ) : (
                      "Predict Reservations"
                    )}
                  </Button>
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
                          <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            className="h-8 w-8 text-muted-foreground/50"
                          >
                            <path d="M9 17H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-4" />
                            <path d="m12 15 5 6H7z" />
                          </svg>
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
            <Card className="shadow-md border-0 bg-white">
              <CardHeader>
                <CardTitle className="text-lg">Latency Benchmark</CardTitle>
                <CardDescription>
                  Measure pure model inference time over 1,000 iterations.
                  This is the time spent in <code className="text-xs bg-muted px-1 py-0.5 rounded">model.predict()</code> only
                  &mdash; excludes HTTP overhead, serialization, and network latency.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <Button
                  onClick={handleBenchmark}
                  disabled={benchLoading || !health}
                  className="rounded-full shadow-lg shadow-primary/25"
                  size="lg"
                >
                  {benchLoading ? (
                    <span className="flex items-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                          fill="none"
                        />
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        />
                      </svg>
                      Running...
                    </span>
                  ) : (
                    "Run Benchmark"
                  )}
                </Button>

                {benchLoading && (
                  <Progress value={benchProgress} className="h-1.5" />
                )}

                {benchmarkResult && (
                  <div className="grid gap-4 sm:grid-cols-4">
                    {[
                      {
                        label: "Average",
                        value: benchmarkResult.avg_latency_ms,
                        highlight: true,
                        desc: "Arithmetic mean of all predictions. Use for capacity planning.",
                      },
                      {
                        label: "p50 (median)",
                        value: benchmarkResult.p50_latency_ms,
                        highlight: false,
                        desc: "Half of predictions are faster than this. Robust to outliers.",
                      },
                      {
                        label: "p95",
                        value: benchmarkResult.p95_latency_ms,
                        highlight: false,
                        desc: "95% of predictions are faster. Typical SLO target.",
                      },
                      {
                        label: "p99",
                        value: benchmarkResult.p99_latency_ms,
                        highlight: false,
                        desc: "Tail latency. A large p95\u2013p99 gap signals instability.",
                      },
                    ].map(({ label, value, highlight, desc }) => (
                      <Tooltip key={label}>
                        <TooltipTrigger className="w-full text-left">
                          <Card
                            className={`border h-full ${highlight ? "border-primary/20 bg-primary/5" : ""}`}
                          >
                            <CardContent className="pt-6 text-center">
                              <div
                                className={`text-3xl font-bold tabular-nums ${highlight ? "text-primary" : ""}`}
                              >
                                {value.toFixed(2)}
                              </div>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {label}
                              </p>
                              <p className="text-[10px] text-muted-foreground/60">
                                ms
                              </p>
                            </CardContent>
                          </Card>
                        </TooltipTrigger>
                        <TooltipContent side="bottom" className="max-w-[200px] text-center">
                          {desc}
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </div>
                )}

                {benchmarkResult && (
                  <p className="text-xs text-muted-foreground text-center">
                    {benchmarkResult.n_iterations.toLocaleString()} iterations
                    using model v{benchmarkResult.model_version}
                  </p>
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
