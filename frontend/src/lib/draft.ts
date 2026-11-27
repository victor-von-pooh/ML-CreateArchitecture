// 保存していない定義の下書き (ブラウザの localStorage に画面ごとに残す)

import { useEffect, useRef } from 'react'

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

/**
 * この画面の下書きを書く (書けない場合は古い下書きを残さないように消す)
 *
 * @param draft - 書く下書き
 * @param tab - 書く場所の印 (既定はこの画面の印)
 */
export function writeDraft(draft: Omit<Draft, 'tab'>, tab = tabId()): void {
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

/**
 * 保存していない変更を, 少し待ってから下書きに書く. 部品が外されるときと `flushKey` が
 * 替わるときは待たずに書く
 *
 * 殻の中で段階を移ると部品が外され, ページを閉じないので `beforeunload` の確かめが出ない.
 * 殻でプロジェクトを替えたときも, そのまま段階を移ることがある. 待つ間の変更を黙って
 * 捨てないように, そのときに書きかけを書く. 前の下書きを戻すかを聞いている間 (`asking` が
 * null でない) は待って書かず, 外されるときと `flushKey` が替わるときだけ書く (聞いている
 * 下書きがこの画面のものなら, 上書きせずに別の画面の下書きとして書く)
 *
 * @param dirty - 保存していない変更があるか
 * @param next - 書く下書き (dirty が false の場合は使わない)
 * @param asking - 戻すかを聞いている下書き (聞いていなければ null)
 * @param flushKey - 替わったときに待たずに書く値 (殻で選んだ問題の設定)
 * @param delay - 待つ時間 (ミリ秒)
 */
export function useDraftWriter(
  dirty: boolean,
  next: Omit<Draft, 'tab' | 'time'>,
  asking: Draft | null,
  flushKey: unknown = null,
  delay = 1000,
): void {
  // まだ書いていない下書きと, 聞いている下書き (書くものが無ければ null)
  const pending = useRef<Omit<Draft, 'tab' | 'time'> | null>(null)
  const askingRef = useRef<Draft | null>(asking)
  const { spec, fileName, saveName, ownFile } = next
  // 聞いている下書きがこの画面のものの間に書く, 別の画面の下書きの印 (書いていなければ null)
  const aside = useRef<string | null>(null)
  // まだ書いていない下書きを待たずに書く
  const flush = useRef(() => {
    const item = pending.current
    if (item === null) return
    pending.current = null
    const shown = askingRef.current
    let tab = tabId()
    if (shown !== null && shown.tab === tab) {
      aside.current ??= `${tab}:${Date.now()}`
      tab = aside.current
      writeDraft({ ...item, time: new Date().toISOString() }, tab)
      return
    }
    // この画面の印で書いたので, 聞いている間に別に書いた下書きは要らない
    writeDraft({ ...item, time: new Date().toISOString() }, tab)
    dropAside()
  })
  // 聞き終えて今の変更をこの画面の印で扱うようになったら, 別に書いた下書きを消す
  const dropAside = () => {
    if (aside.current === null) return
    clearDraft(aside.current)
    aside.current = null
  }
  useEffect(() => {
    askingRef.current = asking
    // 変更を元に戻して保存した時点と同じになった場合は, 下書きも消す (前の
    // 画面の下書きを戻すかを聞いている間は消さない)
    if (!dirty) {
      pending.current = null
      if (asking === null) {
        clearDraft()
        dropAside()
      }
      return
    }
    const item = { spec, fileName, ...(saveName ? { saveName } : {}), ownFile }
    pending.current = item
    // 前の下書きを戻すかを聞いている間は, 前の下書きを上書きしない
    if (asking !== null) return
    const timer = window.setTimeout(() => {
      writeDraft({ ...item, time: new Date().toISOString() })
      dropAside()
      pending.current = null
    }, delay)
    return () => window.clearTimeout(timer)
  }, [dirty, spec, fileName, saveName, ownFile, asking, delay])
  // flushKey が替わったときは, まだ書いていない変更を待たずに書く (最初は書かない)
  const lastKey = useRef(flushKey)
  useEffect(() => {
    if (Object.is(lastKey.current, flushKey)) return
    lastKey.current = flushKey
    flush.current()
  }, [flushKey])
  // 外されるときも, まだ書いていない変更を待たずに書く
  useEffect(() => {
    const write = flush.current
    return () => write()
  }, [])
}
