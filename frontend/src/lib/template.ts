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
