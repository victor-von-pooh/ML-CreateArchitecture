import { createContext, useContext } from 'react'

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
export type ExperimentList = Schemas['ExperimentList']
export type ExperimentEntry = Schemas['ExperimentEntry']
export type ExperimentSpec = Schemas['ExperimentSpec']
export type TemplateMatch = Schemas['TemplateMatch']
export type TemplateAdapted = Schemas['TemplateAdapted']

/** API がエラーを返したときの例外 (detail は文でない detail の中身. 文なら null) */
export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(status: number, message: string, detail: unknown = null) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

/** 骨組みの約束に合わない所の 1 件 (ML-Quest の check_model_contract. id は直す場所) */
export type ContractFinding = { level: string; id: string; message: string }

/** 保存の 409 の detail が骨組みの約束に合わない所なら, 文, 直す場所, 知らせを返す */
export function contractDetail(
  error: unknown,
): { message: string; findings: ContractFinding[]; notes: string[] } | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null
  const detail = error.detail
  if (!detail || typeof detail !== 'object' || !('findings' in detail)) return null
  const { findings, notes } = detail as { findings: unknown; notes?: unknown }
  if (!Array.isArray(findings)) return null
  return {
    message: error.message,
    findings: findings.map((item: unknown) => {
      const record = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
      return {
        level: String(record.level ?? 'error'),
        id: String(record.id ?? ''),
        message: String(record.message ?? ''),
      }
    }),
    notes: Array.isArray(notes) ? notes.map(String) : [],
  }
}

/** エラーのレスポンスから表示する文と, 文でない detail を取り出す */
async function readError(response: Response): Promise<{ message: string; detail: unknown }> {
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === 'string') return { message: detail, detail: null }
      // 画面の文 message を持つ detail はその文 (他は JSON の文)
      if (detail && typeof detail === 'object' && 'message' in detail) {
        const message = (detail as { message: unknown }).message
        if (typeof message === 'string') return { message, detail }
      }
      return { message: JSON.stringify(detail), detail }
    }
  } catch {
    // JSON でない場合は状態の文を使う
  }
  return { message: `${response.status} ${response.statusText}`, detail: null }
}

/** JSON の API を呼ぶ */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const { message, detail } = await readError(response)
    throw new ApiError(response.status, message, detail)
  }
  // 中身の無い応答 (消した場合の 204) は undefined
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

/**
 * API の頭 (単独では `/api`, 殻の下では `/api/model`) の下に送る呼び出しの一式を作る
 *
 * @param base - API の頭 (末尾の `/` は無し)
 */
export function createApi(base: string) {
  return {
    listUnits: () => request<UnitSpec[]>(`${base}/units`),
    /** 起動の設定で読み込んだ自作の部品のモジュール */
    listModules: () => request<ModulesInfo>(`${base}/modules`),
    /** 確認と形の推定 (infer が false なら確認だけ) */
    check: (spec: Spec, infer = true) =>
      request<CheckResult>(`${base}/check`, {
        method: 'POST',
        body: JSON.stringify({ spec, infer }),
      }),
    exportPython: (spec: Spec, className: string) =>
      request<ExportResult>(`${base}/export`, {
        method: 'POST',
        body: JSON.stringify({ spec, class_name: className }),
      }),
    /** 学習済みの重みのライセンス (重みを読まない場合は null) */
    pretrained: (unit: string, params: Record<string, unknown>) =>
      request<PretrainedInfo | null>(`${base}/pretrained`, {
        method: 'POST',
        body: JSON.stringify({ unit, params }),
      }),
    /** 定義全体を組み立てて外部の重みを読み, ノードごとの報告を返す */
    weights: (spec: Spec) =>
      request<WeightsResult>(`${base}/weights`, {
        method: 'POST',
        body: JSON.stringify({ spec }),
      }),
    listModels: () => request<ModelSummary[]>(`${base}/models`),
    listExamples: () => request<ExampleSummary[]>(`${base}/examples`),
    getExample: (name: string) =>
      request<ModelFile>(`${base}/examples/${encodeURIComponent(name)}`),
    getModel: (name: string) => request<ModelFile>(`${base}/models/${encodeURIComponent(name)}`),
    deleteModel: (name: string) =>
      request<void>(`${base}/models/${encodeURIComponent(name)}`, { method: 'DELETE' }),
    /** データセットのディレクトリから, 入力と出す値の数を読む (ML-EDA) */
    datasetInputs: (path: string, problem: string | null = null) =>
      request<DatasetInputs>(`${base}/dataset/inputs`, {
        method: 'POST',
        body: JSON.stringify({ path, problem }),
      }),
    /** 定義が書き換えられる雛形のままなら, 雛形と引数の今の値と場所 (雛形でなければ null) */
    matchTemplate: (spec: Spec) =>
      request<TemplateMatch | null>(`${base}/templates/match`, {
        method: 'POST',
        body: JSON.stringify({ spec }),
      }),
    /** 雛形のままの定義の引数を ML-ModelControl の adapt_example で書き換える (書かない引数は今の値) */
    adaptTemplate: (spec: Spec, name: string, values: Record<string, number>) =>
      request<TemplateAdapted>(`${base}/templates/adapt`, {
        method: 'POST',
        body: JSON.stringify({ spec, name, values }),
      }),
    /** ML-Pathfinder の置き場の問題の設定 (更新の新しい順. 使えないものは理由付き) */
    listProblems: () => request<ProblemList>(`${base}/problems`),
    /** プロジェクトのディレクトリか問題の設定から, 定義の保存先を求める (name は models/ の下の名前. 空なら既定) */
    projectTarget: (path: string, name: string) =>
      request<ProjectTarget>(`${base}/project/target`, {
        method: 'POST',
        body: JSON.stringify({ path, name }),
      }),
    /**
     * 定義をプロジェクトの保存先に書く (target は確かめた保存先. 決め直した保存先と違うか,
     * 既にあり overwrite が偽なら 409. 骨組みの約束に合わない所があり acceptContract が偽なら,
     * 直す場所を detail に付けて 409)
     */
    saveProjectModel: (
      path: string,
      name: string,
      spec: Spec,
      target: string,
      overwrite: boolean,
      acceptContract = false,
    ) =>
      request<ProjectSaved>(`${base}/project/model`, {
        method: 'PUT',
        body: JSON.stringify({
          path,
          name,
          spec,
          target,
          overwrite,
          accept_contract: acceptContract,
        }),
      }),
    /** プロジェクトの実験と, 学習に使った定義の写しを開けるかの一覧 */
    listExperiments: (path: string) =>
      request<ExperimentList>(`${base}/project/experiments`, {
        method: 'POST',
        body: JSON.stringify({ path }),
      }),
    /** 実験の定義の写しを読む (run と root は一覧の run と置き場) */
    openExperimentSpec: (path: string, run: string, root: string) =>
      request<ExperimentSpec>(`${base}/project/experiment-spec`, {
        method: 'POST',
        body: JSON.stringify({ path, run, root }),
      }),
    saveModel: (name: string, spec: Spec) =>
      request<ModelSummary>(`${base}/models/${encodeURIComponent(name)}`, {
        method: 'PUT',
        body: JSON.stringify({ spec }),
      }),
  }
}

/** API の呼び出しの一式 */
export type Api = ReturnType<typeof createApi>

/** 根の部品が決めた API の頭の呼び出しの一式 (既定は単独の頭 `/api`) */
export const ApiContext = createContext<Api>(createApi('/api'))

/** 根の部品が決めた API の頭の呼び出しの一式を使う */
export function useApi(): Api {
  return useContext(ApiContext)
}
