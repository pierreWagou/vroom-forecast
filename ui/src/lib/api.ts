export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  feast_online: boolean;
}

export interface ReloadResponse {
  status: string;
  previous_version: string;
  current_version: string;
}

export interface BenchmarkResponse {
  n_iterations: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  model_version: string;
  source: string;
  avg_features_ms: number;
  avg_predict_ms: number;
}

/** Predict reservations from raw vehicle features (on-the-fly computation). */
export async function predict(vehicle: VehicleFeatures): Promise<PredictionResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(vehicle),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return res.json();
}

/** Check API health, model status, and feature store availability. */
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export interface StoreInfo {
  offline_store: {
    available: boolean;
    type: string;
    path?: string;
    size_bytes?: number;
    last_modified?: number;
  };
  online_store: {
    available: boolean;
    type: string;
    used_memory_human?: string;
    keys?: number;
    redis_url?: string;
  };
  feature_view: {
    name: string;
    entity: string;
    entity_key: string;
    features: string[];
    label: string;
    ttl_days: number;
  };
}

/** Fetch operational info about offline (Parquet) and online (Redis) stores. */
export async function fetchStoreInfo(): Promise<StoreInfo> {
  const res = await fetch(`${API_URL}/stores`);
  if (!res.ok) throw new Error(`Failed to fetch store info: ${res.statusText}`);
  return res.json();
}

/** Benchmark latency using raw features (on-the-fly computation path). */
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

/** Benchmark latency using the online store (Feast/Redis lookup path). */
export async function benchmarkById(
  vehicleId: number,
  nIterations: number
): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_URL}/benchmark/id`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vehicle_id: vehicleId, n_iterations: nIterations }),
  });
  if (!res.ok) throw new Error(`Benchmark failed: ${res.statusText}`);
  return res.json();
}

/** Hot-reload the champion model from MLflow without downtime. */
export async function reloadModel(): Promise<ReloadResponse> {
  const res = await fetch(`${API_URL}/reload`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Reload failed: ${res.statusText}`);
  return res.json();
}

export interface SaveVehicleResponse {
  vehicle_id: number;
  status: string;
  event_published: boolean;
}

export interface VehicleRecord {
  vehicle_id: number;
  technology: number;
  actual_price: number;
  recommended_price: number;
  num_images: number;
  street_parked: number;
  description: number;
  source: string;
  num_reservations: number | null;
}

/** Save a new vehicle to the catalog and trigger feature materialization. */
export async function saveVehicle(vehicle: VehicleFeatures): Promise<SaveVehicleResponse> {
  const res = await fetch(`${API_URL}/vehicles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(vehicle),
  });
  if (!res.ok) throw new Error(`Save failed: ${res.statusText}`);
  return res.json();
}

/** List all vehicles (fleet + new arrivals) with reservation counts. */
export async function listVehicles(): Promise<VehicleRecord[]> {
  const res = await fetch(`${API_URL}/vehicles`);
  if (!res.ok) throw new Error(`Failed to list vehicles: ${res.statusText}`);
  return res.json();
}

/** Delete a new-arrival vehicle from the catalog. */
export async function deleteVehicle(vehicleId: number): Promise<void> {
  const res = await fetch(`${API_URL}/vehicles/${vehicleId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to delete vehicle: ${res.statusText}`);
}

/** Predict reservations for a vehicle by ID using online store features. */
export async function predictById(vehicleId: number): Promise<PredictionResponse> {
  const res = await fetch(`${API_URL}/predict/id`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vehicle_id: vehicleId }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? res.statusText;
    throw new Error(`[${res.status}] ${detail}`);
  }
  return res.json();
}

export interface ComputedFeatures {
  vehicle_id: number;
  technology: number | null;
  num_images: number | null;
  street_parked: number | null;
  description: number | null;
  price_diff: number | null;
  materialized: boolean;
  store: "offline" | "online" | "none";
}

/** Fetch computed features for a single vehicle (offline store, then online fallback). */
export async function fetchVehicleFeatures(vehicleId: number): Promise<ComputedFeatures> {
  const res = await fetch(`${API_URL}/vehicles/${vehicleId}/features`);
  if (!res.ok) throw new Error(`Failed to fetch features: ${res.statusText}`);
  return res.json();
}

/** Fetch computed features for all vehicles in a single batch call. */
export async function fetchAllVehicleFeatures(): Promise<ComputedFeatures[]> {
  const res = await fetch(`${API_URL}/vehicles/features`);
  if (!res.ok) throw new Error(`Failed to fetch features: ${res.statusText}`);
  return res.json();
}
