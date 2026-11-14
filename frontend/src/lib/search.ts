import type { UnitSpec } from './graph'

/**
 * 検索欄の Enter で置く部品 (使える部品のうち, 名前が一致するもの, 名前の
 * 終わりが一致するもの, 名前に含むもの, 説明に含むものの順)
 */
export function firstMatch(units: UnitSpec[], query: string): UnitSpec | undefined {
  const text = query.trim().toLowerCase()
  if (!text) return undefined
  const usable = units.filter((unit) => unit.available)
  const name = (unit: UnitSpec) => unit.name.toLowerCase()
  return (
    usable.find((unit) => name(unit) === text) ??
    usable.find((unit) => name(unit).endsWith(`.${text}`)) ??
    usable.find((unit) => name(unit).includes(text)) ??
    usable.find((unit) => unit.description.toLowerCase().includes(text))
  )
}
