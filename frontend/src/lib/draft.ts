// 保存していない定義の下書き (ブラウザの localStorage に画面ごとに残す)

import type { Spec } from './graph'

// 保存していない定義の下書きを置くブラウザの保存場所の名前 (画面ごとの下書きの辞書)
const DRAFT_KEY = 'ml-createarchitecture:drafts'
// 画面 (タブ) ごとの印を置く場所 (読み込み直しても同じタブなら同じ印)
const TAB_KEY = 'ml-createarchitecture:tab'
// 下書きを戻せる期間 (それより古いものは出さずに消す)
const DRAFT_DAYS = 14

/** 保存していない定義の下書き (どの画面のものか) */
export interface Draft {
  spec: Spec
  fileName: string
  /** 保存する名前の欄に打った, まだ保存していない名前 (打っていなければ無し) */
  saveName?: string
  ownFile: string | null
  time: string
  tab: string
}

/** この画面 (タブ) の印 */
export function tabId(): string {
  try {
    let id = window.sessionStorage.getItem(TAB_KEY)
    if (!id) {
      id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
      window.sessionStorage.setItem(TAB_KEY, id)
    }
    return id
  } catch {
    return 'tab'
  }
}

/** 下書きの辞書を読む (読めない場合は空. 形の合わない下書きと古い下書きは除く) */
function readDrafts(): Record<string, Draft> {
  try {
    const value = JSON.parse(window.localStorage.getItem(DRAFT_KEY) ?? '{}') as Record<
      string,
      Partial<Draft>
    >
    const limit = Date.now() - DRAFT_DAYS * 24 * 60 * 60 * 1000
    const drafts: Record<string, Draft> = {}
    for (const [tab, item] of Object.entries(value ?? {})) {
      const spec = item?.spec as Partial<Spec> | undefined
      const time = Date.parse(String(item?.time))
      if (
        spec &&
        typeof spec === 'object' &&
        spec.nodes &&
        typeof spec.nodes === 'object' &&
        spec.inputs &&
        typeof spec.inputs === 'object' &&
        typeof item.fileName === 'string' &&
        (item.ownFile === null || typeof item.ownFile === 'string') &&
        (item.saveName === undefined || typeof item.saveName === 'string') &&
        Number.isFinite(time) &&
        time > limit
      ) {
        drafts[tab] = { ...(item as Draft), tab }
      }
    }
    return drafts
  } catch {
    return {}
  }
}

/** 戻すかを聞く下書き (この画面のもの. 無ければ他の画面の一番新しいもの) */
export function readDraft(): Draft | null {
  const drafts = readDrafts()
  const own = drafts[tabId()]
  if (own) return own
  const others = Object.values(drafts).sort((a, b) => b.time.localeCompare(a.time))
  return others[0] ?? null
}

/** この画面の下書きを書く (書けない場合は古い下書きを残さないように消す) */
export function writeDraft(draft: Omit<Draft, 'tab'>): void {
  const tab = tabId()
  try {
    window.localStorage.setItem(
      DRAFT_KEY,
      JSON.stringify({ ...readDrafts(), [tab]: { ...draft, tab } }),
    )
  } catch {
    clearDraft(tab)
  }
}

/** 下書きを消す (指定しなければこの画面のもの) */
export function clearDraft(tab = tabId()): void {
  try {
    const drafts = readDrafts()
    if (!(tab in drafts)) return
    delete drafts[tab]
    window.localStorage.setItem(DRAFT_KEY, JSON.stringify(drafts))
  } catch {
    // 保存場所が使えない場合は何もしない
  }
}
