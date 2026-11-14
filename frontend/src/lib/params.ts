import type { ParamSpec } from './graph'

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
