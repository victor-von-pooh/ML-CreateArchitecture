import type { NodeDef, RemapRule, Spec } from './graph'
import { isUnitValue } from './params'

/** 定義の版 2 で足したノードの項目 (パラメータの状態) */
export const STATE_KEYS = ['train', 'eval_mode', 'zero_init', 'weights', 'share'] as const
export type StateKey = (typeof STATE_KEYS)[number]

/** 外部の重みの出どころの種類 */
export type WeightsSource = 'none' | 'path' | 'hub' | 'unit'

/** 定義 (本体か部分モデル) のノードに版 2 の項目があるか */
export function usesState(spec: Spec): boolean {
  const graphs = [spec, ...Object.values(spec.subgraphs ?? {})]
  return graphs.some((graph) =>
    Object.values(graph.nodes ?? {}).some((node) => STATE_KEYS.some((key) => key in node)),
  )
}

/** 定義 (本体か部分モデル) のノードの引数に, 版 3 の項目 (部品と引数の辞書) があるか */
export function usesUnitParams(spec: Spec): boolean {
  const graphs = [spec, ...Object.values(spec.subgraphs ?? {})]
  return graphs.some((graph) =>
    Object.values(graph.nodes ?? {}).some((node) =>
      Object.values(node.params ?? {}).some(isUnitValue),
    ),
  )
}

/**
 * 版 2 の項目を使う定義の版を 2 以上, 版 3 の項目 (引数に書いた部品) を使う
 * 定義の版を 3 以上にしたもの (使わなければそのまま. 保存では ML-ModelControl
 * が書ける最も低い版に直す)
 */
export function withStateVersion(spec: Spec): Spec {
  const needed = usesUnitParams(spec) ? 3 : usesState(spec) ? 2 : 1
  if (spec.format_version >= needed) return spec
  return { ...spec, format_version: needed }
}

/** ノードの状態の項目を書き換えた (undefined なら消した) 定義 */
export function setNodeState<K extends StateKey>(
  spec: Spec,
  name: string,
  key: K,
  value: NodeDef[K] | undefined,
): Spec {
  const node = spec.nodes[name]
  if (!node) return spec
  const next: NodeDef = { ...node }
  if (value === undefined) delete next[key]
  else next[key] = value
  return withStateVersion({ ...spec, nodes: { ...spec.nodes, [name]: next } })
}

/** ノードの状態の項目をまとめて書き換えた (undefined なら消した) 定義 (1 回の変更にするため) */
export function setNodeStates(
  spec: Spec,
  name: string,
  patch: { [K in StateKey]?: NodeDef[K] | undefined },
): Spec {
  const node = spec.nodes[name]
  if (!node) return spec
  const next: NodeDef = { ...node }
  for (const [key, value] of Object.entries(patch) as [StateKey, unknown][]) {
    if (value === undefined) delete next[key]
    else (next as unknown as Record<string, unknown>)[key] = value
  }
  return withStateVersion({ ...spec, nodes: { ...spec.nodes, [name]: next } })
}

/** カンマか空白で区切ったパラメータの名前のリスト */
export function parseNames(text: string): string[] {
  return text
    .split(/[,\s]+/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
}

/** パラメータごとの共有の文字 (1 行に「名前 = ノード.名前」) を辞書に. 読めない行は誤り */
export function parseShareLines(text: string): Record<string, string> | string {
  const result: Record<string, string> = {}
  const lines = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
  for (const line of lines) {
    const match = /^([^=\s]+)\s*=\s*([A-Za-z_][A-Za-z0-9_]*\.[^\s]+)$/.exec(line)
    if (!match) return `「${line}」は「weight = embed.weight」の形にしてください`
    result[match[1]!] = match[2]!
  }
  if (Object.keys(result).length === 0) return '1 行以上書いてください'
  return result
}

/** パラメータごとの共有の辞書を, 1 行に「名前 = ノード.名前」の文字に */
export function formatShareLines(share: Record<string, string>): string {
  return Object.entries(share)
    .map(([key, ref]) => `${key} = ${ref}`)
    .join('\n')
}

/** 外部の重みの出どころの種類 */
export function weightsSource(weights: NodeDef['weights']): WeightsSource {
  if (!weights) return 'none'
  if ('hub' in weights) return 'hub'
  if ('unit' in weights) return 'unit'
  return 'path'
}

/** 出どころの種類を変えた外部の重み (付け替え, 入れ物のキー, strict は残す) */
export function switchWeightsSource(
  weights: NodeDef['weights'],
  source: WeightsSource,
): NodeDef['weights'] {
  if (source === 'none') return undefined
  const { remap, state_key, strict } = weights ?? {}
  const kept = {
    ...(remap ? { remap } : {}),
    ...(state_key ? { state_key } : {}),
    ...(strict === false ? { strict } : {}),
  }
  if (source === 'path') return { path: weights?.path ?? '', ...kept }
  if (source === 'hub') return { hub: weights?.hub ?? '', file: weights?.file ?? '', ...kept }
  return { unit: weights?.unit ?? '', params: weights?.params ?? {}, ...kept }
}

/** 付け替えの 1 行の種類 */
export function ruleKind(rule: RemapRule): 'prefix' | 'pattern' {
  return rule.pattern !== undefined ? 'pattern' : 'prefix'
}

/** 付け替えの 1 行の種類を変えたもの (書いた文字は残す) */
export function switchRuleKind(rule: RemapRule, kind: 'prefix' | 'pattern'): RemapRule {
  if (kind === ruleKind(rule)) return rule
  if (kind === 'pattern') return { pattern: rule.from ?? '', to: rule.to }
  return { from: rule.pattern ?? '', to: rule.to }
}

/** 共有 (share) の元のノードの名前 (共有しなければ空) */
export function shareSources(share: NodeDef['share']): string[] {
  if (share === undefined) return []
  if (typeof share === 'string') return [share]
  return Object.values(share).map((ref) => ref.split('.')[0] ?? '')
}
