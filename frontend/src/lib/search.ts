import type { UnitSpec } from './graph'

/**
 * 検索欄の Enter で置く部品 (使える部品のうち, 名前が一致するもの, 名前の
 * 終わりが一致するもの, よく使う呼び方が一致するもの, 名前に含むもの,
 * よく使う呼び方に含むもの, 説明に含むものの順)
 */
export function firstMatch(units: UnitSpec[], query: string): UnitSpec | undefined {
  const text = query.trim().toLowerCase()
  if (!text) return undefined
  const usable = units.filter((unit) => unit.available)
  const name = (unit: UnitSpec) => unit.name.toLowerCase()
  return (
    usable.find((unit) => name(unit) === text) ??
    usable.find((unit) => name(unit).endsWith(`.${text}`)) ??
    usable.find((unit) => aliasesOf(unit).includes(text)) ??
    usable.find((unit) => name(unit).includes(text)) ??
    usable.find((unit) => aliasesOf(unit).some((alias) => alias.includes(text))) ??
    usable.find((unit) => unit.description.toLowerCase().includes(text))
  )
}

// よく使う呼び方 (他のライブラリの関数の名前など) と, それに当たる部品
const ALIASES: Record<string, string[]> = {
  'pyg.MeanAggregation': ['global_mean_pool', 'global mean pool', 'readout', 'graph pooling'],
  'pyg.MaxAggregation': ['global_max_pool', 'global max pool', 'readout', 'graph pooling'],
  'pyg.SumAggregation': ['global_add_pool', 'global add pool', 'global sum pool', 'readout'],
  'nn.Linear': ['dense', 'fully connected', 'fc'],
  'nn.MultiheadAttention': ['attention', 'self attention', 'mha'],
  'op.Add': ['residual', 'skip connection'],
  'op.Concat': ['cat', 'concatenate'],
  'op.Mean': ['mean pooling', 'average'],
  'op.Max': ['max pooling', 'max pool', 'global max pooling', 'maximum'],
  'nn.Flatten': ['reshape to vector'],
}

/** 部品のよく使う呼び方 (小文字) */
export function aliasesOf(unit: UnitSpec): string[] {
  return ALIASES[unit.name] ?? []
}

/** 部品が検索の文字に当たるか (名前, 説明, よく使う呼び方) */
export function matchesQuery(unit: UnitSpec, text: string): boolean {
  return (
    !text ||
    unit.name.toLowerCase().includes(text) ||
    unit.description.toLowerCase().includes(text) ||
    aliasesOf(unit).some((alias) => alias.includes(text))
  )
}
