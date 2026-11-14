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
  return (await response.json()) as T
}

export const api = {
  listUnits: () => request<UnitSpec[]>('/api/units'),
  check: (spec: Spec) =>
    request<CheckResult>('/api/check', { method: 'POST', body: JSON.stringify({ spec }) }),
  exportPython: (spec: Spec, className: string) =>
    request<ExportResult>('/api/export', {
      method: 'POST',
      body: JSON.stringify({ spec, class_name: className }),
    }),
  listModels: () => request<ModelSummary[]>('/api/models'),
  getModel: (name: string) => request<ModelFile>(`/api/models/${encodeURIComponent(name)}`),
  saveModel: (name: string, spec: Spec) =>
    request<ModelSummary>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify({ spec }),
    }),
}
