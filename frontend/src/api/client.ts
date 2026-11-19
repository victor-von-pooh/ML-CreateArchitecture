import type { components } from './schema'
import type { Spec, UnitSpec } from '../lib/graph'

// API の型 (バックエンドの OpenAPI から生成)
type Schemas = components['schemas']
export type Issue = Schemas['Issue']
export type NodeResult = Schemas['NodeResult']
export type CheckResult = Schemas['CheckResult']
export type ModelSummary = Schemas['ModelSummary']
export type ModelFile = Schemas['ModelFile']
export type ExportResult = Schemas['ExportResult']
export type ExampleSummary = Schemas['ExampleSummary']
export type PretrainedInfo = Schemas['PretrainedInfo']
export type WeightsResult = Schemas['WeightsResult']
export type WeightsReport = Schemas['WeightsReport']
export type ModulesInfo = Schemas['ModulesInfo']

/** API がエラーを返したときの例外 */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** エラーのレスポンスから表示する文を取り出す */
async function readError(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return detail
      return JSON.stringify(detail)
    }
  } catch {
    // JSON でない場合は状態の文を使う
  }
  return `${response.status} ${response.statusText}`
}

/** JSON の API を呼ぶ */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response))
  }
  // 中身の無い応答 (消した場合の 204) は undefined
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  listUnits: () => request<UnitSpec[]>('/api/units'),
  /** 起動の設定で読み込んだ自作の部品のモジュール */
  listModules: () => request<ModulesInfo>('/api/modules'),
  /** 確認と形の推定 (infer が false なら確認だけ) */
  check: (spec: Spec, infer = true) =>
    request<CheckResult>('/api/check', {
      method: 'POST',
      body: JSON.stringify({ spec, infer }),
    }),
  exportPython: (spec: Spec, className: string) =>
    request<ExportResult>('/api/export', {
      method: 'POST',
      body: JSON.stringify({ spec, class_name: className }),
    }),
  /** 学習済みの重みのライセンス (重みを読まない場合は null) */
  pretrained: (unit: string, params: Record<string, unknown>) =>
    request<PretrainedInfo | null>('/api/pretrained', {
      method: 'POST',
      body: JSON.stringify({ unit, params }),
    }),
  /** 定義全体を組み立てて外部の重みを読み, ノードごとの報告を返す */
  weights: (spec: Spec) =>
    request<WeightsResult>('/api/weights', {
      method: 'POST',
      body: JSON.stringify({ spec }),
    }),
  listModels: () => request<ModelSummary[]>('/api/models'),
  listExamples: () => request<ExampleSummary[]>('/api/examples'),
  getExample: (name: string) => request<ModelFile>(`/api/examples/${encodeURIComponent(name)}`),
  getModel: (name: string) => request<ModelFile>(`/api/models/${encodeURIComponent(name)}`),
  deleteModel: (name: string) =>
    request<void>(`/api/models/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  saveModel: (name: string, spec: Spec) =>
    request<ModelSummary>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify({ spec }),
    }),
}
