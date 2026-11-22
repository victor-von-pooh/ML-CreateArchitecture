import type { Spec } from './graph'

/**
 * 問題の設定の雛形をそのまま開いてよい, 新しく始める定義か
 *
 * 部品, 出力, 部分モデルが無く, 保存していない変更も無いもの (新規の画面か, 何も置かずに開いたもの).
 * 入力の形だけを変えた定義も, 保存していない変更なので含めない (黙って捨てないため)
 */
export function startsFresh(spec: Spec, dirty: boolean): boolean {
  return (
    !dirty &&
    Object.keys(spec.nodes).length === 0 &&
    Object.keys(spec.outputs).length === 0 &&
    Object.keys(spec.subgraphs ?? {}).length === 0
  )
}

/**
 * 問題の設定の受け持ちに合う雛形の扱い
 *
 * - open: そのまま開く (一覧から選び, 新しく始める定義で, 保存先にまだ定義が無い)
 * - offer: 開かずに「雛形を開く」を出す (今の定義か, 保存先の定義を上書きしないため)
 * - null: 雛形が無い (今までどおり)
 */
export function templateAction({
  example,
  chosen,
  fresh,
  exists,
}: {
  /** 雛形の名前 (ML-ModelControl の同梱の例. 無ければ null) */
  example: string | null | undefined
  /** 一覧から問題の設定を選んだか (場所を打って確かめた場合は開かない) */
  chosen: boolean
  /** 今の定義が新しく始める定義か (startsFresh) */
  fresh: boolean
  /** 保存先に定義が既にあるか */
  exists: boolean
}): 'open' | 'offer' | null {
  if (!example) return null
  return chosen && fresh && !exists ? 'open' : 'offer'
}

/** 雛形を開かなかった理由 (開く場合は null) */
export function templateReason({
  fresh,
  dirty,
  exists,
}: {
  /** 今の定義が新しく始める定義か (startsFresh) */
  fresh: boolean
  /** 保存していない変更があるか */
  dirty: boolean
  /** 保存先に定義が既にあるか */
  exists: boolean
}): string | null {
  if (!fresh) {
    return dirty ? '保存していない変更があるので開いていません' : '今の定義があるので開いていません'
  }
  return exists ? '保存先に定義が既にあるので開いていません' : null
}

/**
 * 「雛形を開く」を押したときに確かめる文 (確かめずに開く場合は null)
 *
 * 保存していない変更だけでなく, 新しく始める定義でない (部品などがある) 場合も確かめる
 * (雛形で置き換えると, 今の定義は履歴ごと消えるため)
 */
export function templateConfirm({
  fresh,
  dirty,
}: {
  fresh: boolean
  dirty: boolean
}): string | null {
  if (dirty) return '保存していない変更を捨てて, 雛形を開きますか?'
  if (!fresh) return '今の定義を雛形で置き換えますか?'
  return null
}
