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
