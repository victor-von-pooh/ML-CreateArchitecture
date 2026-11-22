import { SUBGRAPH_PREFIX, type ParamSpec, type UnitSpec } from './graph'

/** 入力欄の文字を, 引数の種類に合わせた値にする (空欄は未設定) */
export function parseValue(param: ParamSpec, text: string): unknown {
  const trimmed = text.trim()
  if (trimmed === '') return undefined
  if (param.kind === 'int' && /^-?\d+$/.test(trimmed)) return Number(trimmed)
  if (param.kind === 'float' && trimmed !== '' && !Number.isNaN(Number(trimmed))) {
    return Number(trimmed)
  }
  if (param.kind === 'str') return trimmed
  // 種類の決まらない引数は JSON として読み, 読めなければ文字
  try {
    return JSON.parse(trimmed) as unknown
  } catch {
    return trimmed
  }
}

/** 値を入力欄の文字にする */
export function formatValue(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  return JSON.stringify(value)
}

/** 形の入力 ([4, 16] か 4, 16) を整数のリストにする (読めない場合は null) */
export function parseShape(value: unknown): number[] | null {
  // [4, 16] と (4, 16) (Python の組) と 4, 16 を読む. 括弧の閉じ忘れは読まない
  const text = String(value ?? '').trim()
  if (!Array.isArray(value)) {
    const open = text[0] ?? ''
    const close = text[text.length - 1] ?? ''
    const pairs: Record<string, string> = { '[': ']', '(': ')' }
    if ((open in pairs || close === ']' || close === ')') && pairs[open] !== close) {
      return null
    }
  }
  const items = Array.isArray(value)
    ? value
    : text
        .replace(/^[[(]|[\])]$/g, '')
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
        .map(Number)
  if (items.length === 0) return null
  return items.every((item) => Number.isInteger(item) && (item as number) >= 0)
    ? (items as number[])
    : null
}

/** 入力欄の文字が引数の種類と範囲に合うかを確かめる (合わない場合は理由) */
export function checkValue(param: ParamSpec, text: string): string | null {
  const trimmed = text.trim()
  if (trimmed === '') return param.required ? '必須の引数です' : null
  const value = parseValue(param, trimmed)
  // null は値を決めない (None) 指定
  if (value === null) return param.required ? '必須の引数です' : null
  if (param.kind === 'int' && !Number.isInteger(value)) return '整数を入れてください'
  if (param.kind === 'float' && (typeof value !== 'number' || !Number.isFinite(value))) {
    return '有限の数を入れてください'
  }
  if (typeof value === 'number') {
    if (param.minimum !== null && value < param.minimum)
      return `${param.minimum} 以上にしてください`
    if (param.maximum !== null && value > param.maximum)
      return `${param.maximum} 以下にしてください`
  }
  return null
}

/** 部品を受け取る引数 (種類 unit) に書く値. 登録簿の部品の名前と, その引数 (定義の版 3) */
export interface UnitValue {
  unit: string
  params?: Record<string, unknown>
}

/** 引数に書く部品の入れ子の深さの上限 (ノードの引数に書いた部品が 1 段目. ML-ModelControl と同じ) */
export const UNIT_DEPTH = 10

/** 引数に書いた部品の値 ({unit, params} の辞書) か */
export function isUnitValue(value: unknown): value is UnitValue {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    typeof (value as { unit?: unknown }).unit === 'string'
  )
}

/**
 * 部品を受け取る引数に書ける部品の名前. 引数の候補 (suggestions) を先に,
 * 残りを名前の順に返す. 部分モデル, 使えない部品, 学習済みの重みを読む部品は
 * 書けないので除く (ML-ModelControl の決まり)
 */
export function unitOptions(
  units: Iterable<UnitSpec>,
  suggestions: unknown[] | null | undefined,
): { suggested: string[]; others: string[] } {
  const usable = new Set(
    [...units]
      .filter(
        (unit) =>
          unit.available &&
          !unit.name.startsWith(SUBGRAPH_PREFIX) &&
          (unit.weights_params?.length ?? 0) === 0,
      )
      .map((unit) => unit.name),
  )
  const suggested = (suggestions ?? []).map(String).filter((name) => usable.has(name))
  const others = [...usable].filter((name) => !suggested.includes(name)).sort()
  return { suggested, others }
}

/** 部品を選び直した値 (空は未設定. 同じ部品なら書いた引数を残し, 別の部品なら引数を空に) */
export function pickUnit(value: unknown, unit: string): UnitValue | undefined {
  if (unit === '') return undefined
  if (isUnitValue(value) && value.unit === unit) return value
  return { unit }
}

/** 引数に書いた部品の, 引数を 1 つ変えた (undefined なら消した) 値 (引数が空なら params を除く) */
export function setUnitParam(value: UnitValue, key: string, next: unknown): UnitValue {
  const params = { ...(value.params ?? {}) }
  if (next === undefined) delete params[key]
  else params[key] = next
  const { params: _params, ...rest } = value
  return Object.keys(params).length > 0 ? { ...rest, params } : rest
}
