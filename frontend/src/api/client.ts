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
export type DatasetInputs = Schemas['DatasetInputs']
export type ProjectTarget = Schemas['ProjectTarget']
export type ProjectSaved = Schemas['ProjectSaved']
export type ProblemEntry = Schemas['ProblemEntry']
export type ProblemList = Schemas['ProblemList']

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
  /** データセットのディレクトリから, 入力と出す値の数を読む (ML-EDA) */
  datasetInputs: (path: string) =>
    request<DatasetInputs>('/api/dataset/inputs', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  /** ML-Pathfinder の置き場の問題の設定 (更新の新しい順. 使えないものは理由付き) */
  listProblems: () => request<ProblemList>('/api/problems'),
  /** プロジェクトのディレクトリか問題の設定から, 定義の保存先を求める */
  projectTarget: (path: string) =>
    request<ProjectTarget>('/api/project/target', {
      method: 'POST',
      body: JSON.stringify({ path }),
    }),
  /** 定義をプロジェクトの保存先に書く (既にある場合は overwrite が要る. 無ければ 409) */
  saveProjectModel: (path: string, spec: Spec, overwrite: boolean) =>
    request<ProjectSaved>('/api/project/model', {
      method: 'PUT',
      body: JSON.stringify({ path, spec, overwrite }),
    }),
  saveModel: (name: string, spec: Spec) =>
    request<ModelSummary>(`/api/models/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify({ spec }),
    }),
}
