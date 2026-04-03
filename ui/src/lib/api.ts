const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface VehicleFeatures {
  technology: number;
  actual_price: number;
  recommended_price: number;
  num_images: number;
  street_parked: number;
  description: number;
}

export interface PredictionResponse {
  predicted_reservations: number;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model_name: string;
  model_version: string;
  mlflow_uri: string;
}

export interface BenchmarkResponse {
  n_iterations: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  model_version: string;
}

export async function predict(vehicle: VehicleFeatures): Promise<PredictionResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(vehicle),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function benchmark(
  vehicle: VehicleFeatures,
  nIterations: number
): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_URL}/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vehicle, n_iterations: nIterations }),
  });
  if (!res.ok) throw new Error(`Benchmark failed: ${res.statusText}`);
  return res.json();
}
