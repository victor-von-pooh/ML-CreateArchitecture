import type { CheckResult } from '../api/client'

/**
 * 形の推定で失敗したノードと出力 (選ぶときの名前, 表示の名前, エラーの文)
 *
 * 定義に無いもの (確認の結果が届く前に消したものなど) は除く
 */
export function failedNodes(
  check: CheckResult | undefined,
  spec: { nodes: Record<string, unknown>; outputs: Record<string, unknown> },
): { id: string; label: string; error: string }[] {
  // 確認の誤りのあるノードは確認の誤りとして数えるので除く (形の推定でも
  // error になるため)
  const flagged = new Set(
    (check?.issues ?? [])
      .filter((issue) => issue.severity === 'error' && issue.node)
      .map((issue) => issue.node),
  )
  const nodes = Object.entries(check?.nodes ?? {})
    .filter(
      ([name, result]) => result.status === 'error' && name in spec.nodes && !flagged.has(name),
    )
    .map(([name, result]) => ({ id: name, label: name, error: result.error ?? '' }))
  const outputs = Object.entries(check?.outputs ?? {})
    .filter(([name, value]) => {
      const info = value as ShapeInfo | null
      return info?.kind === 'error' && name in spec.outputs
    })
    .map(([name, value]) => ({
      id: `output:${name}`,
      label: `出力 ${name}`,
      error: (value as ShapeInfo).error ?? '',
    }))
  return [...nodes, ...outputs]
}

/** 形の推定の結果 (ML-ModelControl の describe の形) */
export interface ShapeInfo {
  kind: string
  shape?: number[]
  dtype?: string
  items?: ShapeInfo[] | Record<string, ShapeInfo>
  type?: string
  error?: string
}

/** 出力の形を短い文字にする ([4, 16] や (2 個の組) など) */
export function formatShape(value: unknown): string {
  const info = value as ShapeInfo
  if (!info || typeof info !== 'object') return ''
  switch (info.kind) {
    case 'tensor':
      return `[${(info.shape ?? []).join(', ')}]`
    case 'tuple':
      return `(${Array.isArray(info.items) ? info.items.length : 0} 個の組)`
    case 'dict':
      return '{…}'
    case 'none':
      return 'None'
    case 'error':
      return 'エラー'
    default:
      return info.type ?? ''
  }
}

/** パラメータ数を読みやすくする (1,234,567 → 1.23M) */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return '-'
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return String(value)
}
