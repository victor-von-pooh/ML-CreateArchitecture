import { useQuery } from '@tanstack/react-query'
import { ReactFlowProvider, useReactFlow } from '@xyflow/react'
import { PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen } from 'lucide-react'
import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from 'react'

import { api } from './api/client'
import { ErrorBox, Spinner } from './components/ui'
import { Canvas } from './features/editor/Canvas'
import { Inspector } from './features/editor/Inspector'
import { Palette } from './features/editor/Palette'
import { Toolbar } from './features/editor/Toolbar'
import { SubgraphDialog } from './features/editor/SubgraphDialog'
import {
  addNode,
  emptySpec,
  pasteNode,
  SUBGRAPH_PREFIX,
  connectFrom,
  type NodeDef,
  type Spec,
  type UnitSpec,
} from './lib/graph'
import {
  embedSubgraph,
  removeSubgraph,
  subgraphUnits,
  subgraphUsers,
  subgraphView,
  makesCycle,
  placeholderUnits,
  forkSubgraph,
  subgraphOrigin,
  refreshSubgraph,
  usageShapes,
  withInputShapes,
  subgraphCheckTarget,
  writeSubgraph,
} from './lib/subgraph'
import { clearDraft, readDraft, tabId, writeDraft, type Draft } from './lib/draft'
import { useHistory } from './lib/history'
import { arrangeSpec, freeSpot, placeMissing, unitSize } from './lib/layout'

/** 書き換えている途中の値が落ち着いてから使う (確認の回数を抑える) */
function useDebounced<T>(value: T, delay: number): T {
  const [current, setCurrent] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setCurrent(value), delay)
    return () => window.clearTimeout(timer)
  }, [value, delay])
  return current
}

/** モデルの定義を部品の一覧, キャンバス, 詳細で編集する画面 */
export function App() {
  return (
    <ReactFlowProvider>
      <Editor />
    </ReactFlowProvider>
  )
}

/** 編集の画面の本体 (キャンバスの表示範囲を使うので ReactFlowProvider の中) */
function Editor() {
  const units = useQuery({ queryKey: ['units'], queryFn: api.listUnits, staleTime: Infinity })
  // 定義は元に戻す・やり直すができる値で持つ
  const history = useHistory<Spec>(() => emptySpec())
  // 保存する定義 (whole) と, 画面で編集している定義 (spec. 部分モデルを
  // 編集している間はその部分モデル)
  const whole = history.value
  // 最新の定義 (読み込みを待つ操作が, 待った後の定義を使うため)
  const latest = useRef(whole)
  useEffect(() => {
    latest.current = whole
  }, [whole])
  // 編集している部分モデルの並び (入れ子で開いた順. 空なら本体)
  const [editPath, setEditPath] = useState<string[]>([])
  // 元に戻すなどで部分モデルが無くなった場合は, あるところまでに戻す
  const missing = editPath.findIndex((name) => !whole.subgraphs?.[name])
  const validPath = missing < 0 ? editPath : editPath.slice(0, missing)
  if (validPath.length !== editPath.length) setEditPath(validPath)
  const editingName = validPath.at(-1) ?? null
  const setEditing = (name: string | null) => setEditPath(name === null ? [] : [name])
  const spec = useMemo(
    () => (editingName ? subgraphView(whole, editingName) : whole),
    [whole, editingName],
  )
  const setHistory = history.set
  // 部分モデルを編集している間の変更は, 定義の中の部分モデルに書き戻す
  const setSpec = useCallback(
    (next: Spec | ((current: Spec) => Spec), merge?: string) => {
      if (!editingName) {
        setHistory(next, merge)
        return
      }
      setHistory((current) => {
        if (!current.subgraphs?.[editingName]) return current
        const view = subgraphView(current, editingName)
        const value = typeof next === 'function' ? next(view) : next
        return writeSubgraph(current, editingName, value)
      }, merge)
    },
    [editingName, setHistory],
  )
  // 履歴の定義を, 画面で編集している形にしたもの
  const viewOf = useCallback(
    (target: Spec) =>
      editingName && target.subgraphs?.[editingName] ? subgraphView(target, editingName) : target,
    [editingName],
  )
  const [fileName, setFileName] = useState('model')
  // 今の定義を開いた (か保存した) 保存した定義の名前 (上書きの確認に使う)
  const [ownFile, setOwnFile] = useState<string | null>(null)
  // 保存した (か開いた) 時点の定義. 今の定義と比べて, 保存していない変更を判定
  const [saved, setSaved] = useState(() => JSON.stringify(emptySpec()))
  // 前に開いていた画面の, 保存していない定義の下書き (戻すか捨てるかを聞く)
  const [draft, setDraft] = useState<Draft | null>(readDraft)
  // 保存する名前の欄に打った, まだ保存していない名前 (下書きに残す. 開く,
  // 新規, 保存で消す) と, 下書きから戻した回数 (名前の欄を作り直すため)
  const [typedName, setTypedName] = useState<string | null>(null)
  const [restored, setRestored] = useState(0)
  const [selected, setSelected] = useState<string | null>(null)
  const flow = useReactFlow()

  // 登録簿の部品に, 定義ごとの部分モデルの部品 (sub.<名前>) を加えたもの
  const baseUnits = useMemo(
    () => new Map((units.data ?? []).map((unit) => [unit.name, unit])),
    [units.data],
  )
  // (使えない部品と登録されていない部品は, 定義から口を推し量って線を描く)
  const unitsFor = useCallback(
    (target: Spec) => {
      const local = subgraphUnits(target)
      const withLocal =
        local.length === 0
          ? baseUnits
          : new Map([...baseUnits, ...local.map((unit) => [unit.name, unit] as const)])
      const placeholders = placeholderUnits(target, withLocal)
      if (placeholders.length === 0) return withLocal
      return new Map([...withLocal, ...placeholders.map((unit) => [unit.name, unit] as const)])
    },
    [baseUnits],
  )
  // 定義が落ち着いたら確認と形の推定 (前の結果は次が届くまで表示)
  // 部分モデルを編集している間は, その部分モデルを 1 つの定義として確かめる
  // (自分自身を除き, 中で使う部分モデルだけを持たせる)
  // 部分モデルを編集している間は, 本体も確かめて, 本体の誤り (口の名前を
  // 変えた場合など) を知らせ, 本体で使う所に入る形を部分モデルの入力の形に
  // する (本体の部分モデルを開いた場合)
  const debouncedWhole = useDebounced(whole, 500)
  // (形を使うのは本体の部分モデルを開いた場合だけなので, 入れ子では確認だけ)
  const inferWhole = validPath.length === 1
  const wholeCheck = useQuery({
    queryKey: ['check-whole', debouncedWhole, inferWhole],
    queryFn: () => api.check(debouncedWhole, inferWhole),
    enabled: editingName !== null,
    staleTime: Infinity,
  })
  const usage = useMemo(
    () =>
      editingName && validPath.length === 1
        ? usageShapes(whole, wholeCheck.data, editingName, unitsFor(whole))
        : null,
    [editingName, validPath.length, whole, wholeCheck.data, unitsFor],
  )
  const target = useMemo(
    () =>
      editingName
        ? withInputShapes(subgraphCheckTarget(spec, editingName), usage?.shapes ?? {})
        : spec,
    [spec, editingName, usage],
  )
  const debounced = useDebounced(target, 500)
  const check = useQuery({
    queryKey: ['check', debounced],
    queryFn: () => api.check(debounced),
    placeholderData: (previous) => previous,
    staleTime: Infinity,
  })

  const wholeErrors = editingName
    ? (wholeCheck.data?.issues ?? []).filter((issue) => issue.severity === 'error').length
    : 0

  const unitMap = useMemo(() => unitsFor(spec), [unitsFor, spec])
  // 部分モデルを編集している間は, 置くと循環になるもの (自分自身と, 中で
  // 自分を使う部分モデル) を部品の一覧に出さない
  const paletteUnits = useMemo(
    () => [
      ...subgraphUnits(spec).filter(
        (unit) => !editingName || !makesCycle(spec.subgraphs, editingName, unit.name),
      ),
      ...(units.data ?? []),
    ],
    [spec, units.data, editingName],
  )
  const [importing, setImporting] = useState(false)
  // 部品を探す欄に移る要求 (欄を開いた後に移るため, 回数で表す. 部品の一覧の
  // 読み込みの間に押した場合は, 読み込んだ後に移る)
  const [searchRequest, setSearchRequest] = useState(0)
  useEffect(() => {
    if (searchRequest === 0) return
    const search = document.querySelector<HTMLInputElement>('aside[aria-label="部品の一覧"] input')
    search?.focus()
    search?.select()
  }, [searchRequest, units.isPending])
  // 左右の欄を開いているか (狭い画面では最初から畳む)
  const [panels, setPanels] = useState(() => {
    const wide = typeof window === 'undefined' || window.innerWidth >= 1200
    return { left: wide, right: wide }
  })
  // 保存した知らせ (少しの間だけ上の操作に出す)
  const [savedNotice, setSavedNotice] = useState('')
  // 部品の一覧に出す, 部分モデルの操作の結果の知らせ
  const [notice, setNotice] = useState<{ text: string } | null>(null)
  const notify = (text: string) => setNotice({ text })
  const dirty = useMemo(() => JSON.stringify(whole) !== saved, [whole, saved])
  // 保存していない変更は, 少し待ってからブラウザに下書きとして残す (閉じたり
  // 読み込み直したりしても失わないように. 保存, 新規, 開く, 捨てるで消す)
  useEffect(() => {
    // 変更を元に戻して保存した時点と同じになった場合は, 下書きも消す (前の
    // 画面の下書きを戻すかを聞いている間は消さない)
    if (!dirty) {
      if (draft === null) clearDraft()
      return
    }
    // 前の下書きを戻すかを聞いている間は, 前の下書きを上書きしない
    if (draft !== null) return
    const timer = window.setTimeout(
      () =>
        writeDraft({
          spec: whole,
          fileName,
          ...(typedName !== null && typedName !== fileName ? { saveName: typedName } : {}),
          ownFile,
          time: new Date().toISOString(),
        }),
      1000,
    )
    return () => window.clearTimeout(timer)
  }, [dirty, whole, fileName, typedName, ownFile, draft])

  // 新しく置くものは, キャンバスに見えている範囲の真ん中の近くで, 他の
  // ノードと重ならない所
  const center = (size = { width: 200, height: 100 }, near?: { x: number; y: number }) => {
    const box = document.querySelector('.react-flow')?.getBoundingClientRect()
    const point = box
      ? flow.screenToFlowPosition({ x: box.left + box.width / 2, y: box.top + box.height / 2 })
      : { x: 300, y: 120 }
    // 見えている範囲 (キャンバスの座標)
    const topLeft = box ? flow.screenToFlowPosition({ x: box.left, y: box.top }) : undefined
    const bottomRight = box ? flow.screenToFlowPosition({ x: box.right, y: box.bottom }) : undefined
    const bounds =
      topLeft && bottomRight
        ? {
            x: topLeft.x,
            y: topLeft.y,
            width: bottomRight.x - topLeft.x,
            height: bottomRight.y - topLeft.y,
          }
        : undefined
    const rects = flow.getNodes().map((node) => ({
      x: node.position.x,
      y: node.position.y,
      width: node.measured?.width ?? 200,
      height: node.measured?.height ?? 100,
    }))
    // 望む位置が見えている範囲の外なら, 見えている範囲の真ん中
    const visible =
      near &&
      (!bounds ||
        (near.x >= bounds.x &&
          near.y >= bounds.y &&
          near.x + size.width <= bounds.x + bounds.width &&
          near.y + size.height <= bounds.y + bounds.height))
    // 望む位置 (選んでいるものの右や複製の元の近く) が見えている範囲の外なら, その位置に置いて
    // 表示をそこへ動かす (真ん中に飛ばさず, つながりの順に並ぶように)
    if (near && !visible) {
      const spot = freeSpot(rects, near, size)
      window.requestAnimationFrame(
        () =>
          void flow.setCenter(spot.x + size.width / 2, spot.y + size.height / 2, {
            duration: 200,
            zoom: flow.getZoom(),
          }),
      )
      return spot
    }
    const wanted = visible
      ? near
      : { x: Math.round(point.x - size.width / 2), y: Math.round(point.y - size.height / 2) }
    return freeSpot(rects, wanted, size, bounds)
  }

  // 保存していない変更がある場合は, 閉じる前に確認
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault()
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  // 選んでいるもの (ノード, 入力, 出力) が定義にあるか
  const has = (target: Spec, id: string) =>
    id in target.nodes ||
    id in target.inputs ||
    (id.startsWith('output:') &&
      (id.slice(7) in target.outputs || id.slice(7) in (target.ui?.outputs ?? {})))

  // 無くなったノードは選んでいないものとする (念のため)
  const current = selected === null || has(spec, selected) ? selected : null

  // 元に戻す・やり直すで選んでいるものが無くなる場合は, 選択を外す
  const { undo, redo, previous, next: following } = history
  // 名前の変更を戻す (やり直す) 場合は, 選択も元の (新しい) 名前に移す
  const reselect = useCallback(
    (previousSpec: Spec) => {
      const target = viewOf(previousSpec)
      if (selected === null || has(target, selected)) return
      // ノード, 入力, 出力 (output: を付けた id) の名前を比べる
      const ids = (item: Spec) => [
        ...Object.keys(item.nodes),
        ...Object.keys(item.inputs),
        ...[...Object.keys(item.outputs), ...Object.keys(item.ui?.outputs ?? {})].map(
          (name) => `output:${name}`,
        ),
      ]
      const now = new Set(ids(spec))
      const then = new Set(ids(target))
      const gone = [...then].filter((id) => !now.has(id))
      const added = [...now].filter((id) => !then.has(id))
      if (gone.length === 1 && added.length === 1 && added[0] === selected) {
        setSelected(gone[0]!)
      } else {
        setSelected(null)
      }
    },
    [selected, spec, viewOf],
  )
  const undoSpec = useCallback(() => {
    if (!previous) return
    reselect(previous)
    undo()
  }, [previous, reselect, undo])
  const redoSpec = useCallback(() => {
    if (!following) return
    reselect(following)
    redo()
  }, [following, reselect, redo])

  // 写したノード (部品, 引数, 呼び出しの設定, 説明)
  const [clipboard, setClipboard] = useState<NodeDef | null>(null)

  // ノードの写しを置いて選ぶ (置き場所は, 見えている範囲の真ん中か元のノードの右下)
  const paste = (source: NodeDef, position: { x: number; y: number }) => {
    if (editingName && makesCycle(spec.subgraphs, editingName, source.unit)) {
      notify(`${source.unit} は sub.${editingName} の中に置けません (互いに使い合うため)`)
      return
    }
    const pasted = pasteNode(spec, source, position)
    setSpec(pasted.spec)
    setSelected(pasted.name)
  }
  const duplicate = (name: string) => {
    const node = spec.nodes[name]
    if (!node) return
    // 元のノードの近くで, 重ならない所
    const size = unitSize(unitMap.get(node.unit))
    paste(node, center(size, { x: (node.ui?.x ?? 0) + 40, y: (node.ui?.y ?? 0) + 40 }))
  }

  // 写す (Ctrl+C), 貼る (Ctrl+V), 複製する (Ctrl+D). 部品のノードを選んで
  // いる場合だけ (それ以外はブラウザの標準の操作)
  const onClipboardKey = useEffectEvent((key: string): boolean => {
    const node = current !== null ? spec.nodes[current] : undefined
    // 文字を選んでいる場合の Ctrl+C は, その文字の写し (ブラウザの標準)
    const text = !(window.getSelection()?.isCollapsed ?? true)
    if (key === 'c' && node && !text) setClipboard(node)
    else if (key === 'v' && clipboard)
      paste(clipboard, center(unitSize(unitMap.get(clipboard.unit))))
    else if (key === 'd' && node && current !== null) duplicate(current)
    else return false
    return true
  })

  // 部品はキャンバスの真ん中に置いて選択
  // 部品のノードを選んでいる場合は, その右 (つなぐ先になることが多い) に置く
  // focus: キーボードで置いた場合 (検索欄の Enter) は詳細の最初の欄に移る
  // at: 部品の一覧からキャンバスに落とした位置 (その場所に置き, 線は引かない)
  const add = (unit: UnitSpec, focus = false, at?: { x: number; y: number }) => {
    if (editingName && makesCycle(spec.subgraphs, editingName, unit.name)) {
      notify(`${unit.name} は sub.${editingName} の中に置けません (互いに使い合うため)`)
      return
    }
    const from = current !== null ? (spec.nodes[current] ?? spec.inputs[current]) : undefined
    const near =
      from?.ui?.x !== undefined && from.ui.y !== undefined
        ? { x: from.ui.x + 280, y: from.ui.y }
        : undefined
    const placed = addNode(spec, unit, at ?? center(unitSize(unit), near))
    const name = Object.keys(placed.nodes).find((key) => !(key in spec.nodes)) ?? null
    // 選んでいる入力かノードがあれば, つなぎ方が決まる場合はそこから線を引く
    const units = new Map(unitMap).set(unit.name, unit)
    const next =
      !at && from && current !== null && name ? connectFrom(placed, units, current, name) : placed
    setSpec(next)
    setSelected(name)
    // キーボードで続けて操作できるように, 詳細の最初の欄に移る (畳んでいれば開く)
    if (!focus) return
    setPanels((current) => ({ ...current, right: true }))
    // 名前の欄ではなく, 引数の最初の欄 (無ければ入力のつなぎ先) に移る
    window.requestAnimationFrame(() => {
      const panel = 'aside[aria-label="詳細"]'
      ;(
        document.querySelector<HTMLElement>(`${panel} [data-first-field] :is(input, select)`) ??
        document.querySelector<HTMLElement>(`${panel} fieldset :is(select, input)`)
      )?.focus()
    })
  }

  // 2 つ以上のノード (本体と部分モデルの中) が使う部分モデルの名前
  // (1 つのノードだけのものにする操作を出すもの)
  const forkable = new Set(
    Object.keys(whole.subgraphs ?? {}).filter(
      (name) =>
        [whole, ...Object.values(whole.subgraphs ?? {})]
          .flatMap((graph) => Object.values(graph.nodes ?? {}))
          .filter((node) => node.unit === `${SUBGRAPH_PREFIX}${name}`).length > 1,
    ),
  )

  // 元の定義を写した部分モデル (写し直せるもの) の部品の名前
  const refreshable = new Set(
    Object.entries(whole.subgraphs ?? {})
      .filter(([, sub]) => subgraphOrigin(sub))
      .map(([name]) => `${SUBGRAPH_PREFIX}${name}`),
  )
  // 部分モデルを, 写した元の定義の今の中身で置き換える (名前はそのまま)
  const refresh = async (name: string) => {
    const origin = subgraphOrigin(whole.subgraphs?.[name])
    if (!origin) return
    try {
      const model =
        origin.kind === 'model'
          ? await api.getModel(origin.name)
          : await api.getExample(origin.name)
      if (origin.kind === 'model' && origin.name === ownFile) {
        notify(`${origin.name} は今開いている定義なので写し直せません`)
        return
      }
      // 読み込みの間に定義が変わっても, 今の定義に対して行う
      const outcome = refreshSubgraph(latest.current, name, model.spec as unknown as Spec, origin)
      if (!outcome) notify(`sub.${name} はもう無いので写し直しませんでした`)
      else if (!outcome.changed) notify(`sub.${name} は ${origin.name} と同じ中身です`)
      else {
        setHistory(outcome.spec)
        notify(`sub.${name} を ${origin.name} の今の中身で写し直しました (元に戻すで戻せます)`)
      }
    } catch (error) {
      notify(
        `sub.${name} を写し直せません: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
  }

  // 部分モデルの編集を始める (path が空なら本体に戻る)
  const openPath = (path: string[]) => {
    setEditPath(path)
    setSelected(null)
    const name = path.at(-1)
    if (name && whole.subgraphs?.[name]) {
      // 位置の無いもの (写した例など) は, 編集を始めるときに位置を書く
      // (履歴に残さず, 保存していない変更にもしない)
      const view = subgraphView(whole, name)
      const placed = placeMissing(view, unitsFor(view))
      if (JSON.stringify(placed) !== JSON.stringify(view)) {
        const next = writeSubgraph(whole, name, placed)
        history.replace(next)
        if (!dirty) setSaved(JSON.stringify(next))
      }
    }
    window.requestAnimationFrame(() => {
      void flow.fitView({ duration: 200, maxZoom: 1, padding: 0.1 })
      // 編集を始めたら戻る操作に, 戻ったらキャンバスに
      document
        .querySelector<HTMLElement>(path.length > 0 ? '[data-edit-back]' : '.react-flow')
        ?.focus()
    })
  }

  // 選んでいるノード (無ければキャンバス) に focus を移す
  const focusSelected = useEffectEvent(() => {
    const node =
      current !== null
        ? document.querySelector<HTMLElement>(`.react-flow__node[data-id="${CSS.escape(current)}"]`)
        : null
    ;(node ?? document.getElementById('canvas'))?.focus()
  })
  // Esc は編集している部分モデルから 1 つ上に戻り, / は部品を探す欄に移る
  const onNavKey = useEffectEvent((key: string): boolean => {
    if (key === 'Escape' && validPath.length > 0) {
      openPath(validPath.slice(0, -1))
      return true
    }
    if (key === '/') {
      // 畳んでいる場合は開いてから (開いた後の描画で) 移る
      setPanels((current) => ({ ...current, left: true }))
      setSearchRequest((count) => count + 1)
      return true
    }
    return false
  })
  // Ctrl (Mac は Cmd) + Z で元に戻し, Shift も押すか Ctrl + Y でやり直す
  // 文字を打つ欄ではブラウザの標準の操作 (文字の取り消し) を使い, 開いた
  // ダイアログの中では定義を戻さない
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      // Ctrl+S (Mac は Cmd+S) は文字の欄の中でも保存 (ブラウザのページの保存の
      // 代わり). 欄の打ちかけの値は, 欄を離れて反映してから保存. ダイアログの
      // 中では何もしない
      const modifier = (event.ctrlKey || event.metaKey) && !event.altKey
      if (modifier && event.key.toLowerCase() === 's' && !event.shiftKey) {
        event.preventDefault()
        if (target?.closest('dialog[open]')) return
        // 保存の後は, 前にいた所に戻る (focus を失わない)
        const before = document.activeElement instanceof HTMLElement ? document.activeElement : null
        before?.blur()
        window.requestAnimationFrame(() => {
          document.querySelector<HTMLButtonElement>('[data-save]:not(:disabled)')?.click()
          if (before?.isConnected) before.focus()
        })
        return
      }
      if (target?.closest('dialog[open], textarea') || target?.isContentEditable) return
      // 詳細の欄の中の Esc は, 欄を離れて選んでいるノード (無ければキャンバス) に戻る
      // 名前の欄には / を書けないので, / は部品を探す欄に移る (置いた後に続けて探せる)
      if (
        event.key === '/' &&
        !event.ctrlKey &&
        !event.metaKey &&
        target instanceof HTMLInputElement &&
        target.dataset.nameField !== undefined
      ) {
        event.preventDefault()
        onNavKey('/')
        return
      }
      if (
        event.key === 'Escape' &&
        !event.isComposing &&
        event.keyCode !== 229 &&
        target?.closest('aside[aria-label="詳細"]') &&
        (target instanceof HTMLInputElement || target instanceof HTMLSelectElement)
      ) {
        event.preventDefault()
        // 打ちかけの文字は捨てて (Esc は取り消し) から欄を離れる
        if (target instanceof HTMLInputElement && target.type !== 'checkbox') {
          target.value = target.dataset.committed ?? target.defaultValue
        }
        target.blur()
        focusSelected()
        return
      }
      // 他で扱ったキーは扱わない (Esc はキャンバスが選択を外すのにも使うので除く)
      if (event.defaultPrevented && event.key !== 'Escape') return
      // 文字を打つ欄ではブラウザの標準 (文字の取り消し). ただし打ちかけの文字
      // が無い欄 (値を反映した後) の Ctrl+Z とやり直しは定義を戻す (欄の文字
      // だけが戻って定義と食い違わないように)
      const pristine =
        target instanceof HTMLInputElement &&
        target.dataset.committed !== undefined &&
        target.value === target.dataset.committed &&
        (event.ctrlKey || event.metaKey) &&
        ['z', 'y'].includes(event.key.toLowerCase())
      if (
        target instanceof HTMLInputElement &&
        !pristine &&
        !['checkbox', 'radio', 'button', 'submit', 'range', 'color'].includes(target.type)
      ) {
        return
      }
      // 修飾キーの無い Esc (部分モデルを 1 つ上に戻る) と / (部品を探す欄へ)
      if (!event.ctrlKey && !event.metaKey && !event.altKey) {
        if (target instanceof HTMLSelectElement) return
        // 押し続けた場合の繰り返しでは, 階層を続けて戻らない
        if (event.repeat) return
        if ((event.key === 'Escape' || event.key === '/') && onNavKey(event.key)) {
          event.preventDefault()
        }
        return
      }
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return
      const key = event.key.toLowerCase()
      if (key === 'z' && !event.shiftKey) undoSpec()
      else if ((key === 'z' && event.shiftKey) || key === 'y') redoSpec()
      else if (event.shiftKey || !['c', 'v', 'd'].includes(key)) return
      // 押し続けた場合の繰り返しでは, 写しを増やさない
      else if (event.repeat) return
      else if (!onClipboardKey(key)) return
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undoSpec, redoSpec])

  if (units.isPending) return <Spinner label="部品の一覧を読み込み中" />
  if (units.error) {
    return (
      <div className="p-6">
        <ErrorBox error={units.error} />
      </div>
    )
  }

  return (
    <>
      <div className="flex h-screen flex-col bg-bg text-ink">
        {/* キーボードで, 上の操作と部品の一覧を飛ばしてキャンバスへ移る */}
        <a
          href="#canvas"
          className="sr-only z-50 rounded-md border border-accent bg-surface text-sm focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:px-3 focus:py-2"
        >
          キャンバスへ移る
        </a>
        <Toolbar
          // 開いた定義が変わったか下書きを戻したら, 保存する名前の欄を作り直す
          key={`${fileName}:${restored}`}
          initialName={typedName ?? undefined}
          onNameChange={setTypedName}
          savedNotice={savedNotice}
          spec={spec}
          units={unitMap}
          whole={whole}
          editing={editingName}
          fileName={fileName}
          dirty={dirty}
          check={check.data}
          checking={check.isFetching || debounced !== target}
          onChange={setSpec}
          onLoad={(name, loaded, source) => {
            clearDraft()
            setDraft(null)
            setTypedName(null)
            // 前の定義の部分モデルの知らせは消す
            setNotice(null)
            setOwnFile(source === 'model' ? name : null)
            // 位置の無いもの (手で書いた定義など) は, 開いたときに一度だけ
            // 段に分けた位置を書く (後の編集で動かないように)
            const placed = placeMissing(loaded, unitsFor(loaded))
            history.reset(placed)
            setEditing(null)
            setSaved(JSON.stringify(placed))
            // 開いた定義の全体が見えるように表示の範囲を合わせる
            window.requestAnimationFrame(
              () => void flow.fitView({ duration: 200, maxZoom: 1, padding: 0.1 }),
            )
            setFileName(name)
            setSelected(null)
          }}
          ownFile={ownFile}
          onDeletedOwn={() => {
            setOwnFile(null)
            setSaved('')
          }}
          onSaved={(name, savedSpec) => {
            clearDraft()
            setTypedName(null)
            if (draft?.tab === tabId()) setDraft(null)
            const errorCount = (check.data?.issues ?? []).filter(
              (issue) => issue.severity === 'error',
            ).length
            setSavedNotice(
              errorCount > 0
                ? `${name} に保存しました (誤り ${errorCount} 件のまま)`
                : `${name} に保存しました`,
            )
            window.setTimeout(() => setSavedNotice(''), 3000)
            setOwnFile(name)
            // 保存した定義の名前を今の定義にも反映 (保存の間の編集は残す)
            history.replace((current) => ({ ...current, name: savedSpec.name }))
            setFileName(name)
            setSaved(JSON.stringify(savedSpec))
          }}
          center={center}
          onSelect={setSelected}
          onArrange={() => {
            setSpec((current) => arrangeSpec(current, unitsFor(current)))
            // 並べ直した全体が見えるように表示の範囲を合わせる
            // (文字が読める倍率より小さくはしない)
            window.requestAnimationFrame(
              () => void flow.fitView({ duration: 200, maxZoom: 1, minZoom: 0.2, padding: 0.1 }),
            )
          }}
          onUndo={previous ? undoSpec : undefined}
          onRedo={following ? redoSpec : undefined}
        />
        {draft && (
          <section
            aria-label="保存していない定義の下書き"
            className="flex flex-wrap items-center gap-2 border-b border-line bg-accent-soft px-3 py-1.5 text-xs"
          >
            <span>
              保存していない定義{' '}
              <span className="font-mono">{draft.saveName || draft.fileName}</span> (
              {new Date(draft.time).toLocaleString()}) の下書きがあります
            </span>
            <button
              type="button"
              onClick={() => {
                if (
                  dirty &&
                  !window.confirm('保存していない今の変更を捨てて, 下書きを戻しますか?')
                ) {
                  return
                }
                let placed: Spec
                try {
                  placed = placeMissing(draft.spec, unitsFor(draft.spec))
                } catch {
                  clearDraft(draft.tab)
                  setDraft(null)
                  return
                }
                // 戻した下書きは, この画面のものとして書き直す
                if (draft.tab !== tabId()) clearDraft(draft.tab)
                history.reset(placed)
                window.requestAnimationFrame(
                  () => void flow.fitView({ duration: 200, maxZoom: 1, padding: 0.1 }),
                )
                setFileName(draft.fileName)
                setTypedName(draft.saveName ?? null)
                setRestored((count) => count + 1)
                setOwnFile(draft.ownFile)
                // 保存していない変更として戻す
                setSaved('')
                setEditPath([])
                setSelected(null)
                setDraft(null)
              }}
              className="rounded border border-line bg-surface px-2 py-0.5 font-medium hover:bg-surface-2"
            >
              下書きを開く
            </button>
            <button
              type="button"
              onClick={() => {
                clearDraft(draft.tab)
                setDraft(null)
              }}
              className="rounded px-2 py-0.5 text-ink-2 hover:bg-surface-2"
            >
              下書きを捨てる
            </button>
          </section>
        )}
        <div
          className="grid min-h-0 flex-1"
          style={{
            // 畳んだ欄 (hidden) は格子に並ばないので, その列も作らない
            gridTemplateColumns: [
              panels.left ? '16rem' : null,
              'minmax(0,1fr)',
              panels.right ? '20rem' : null,
            ]
              .filter(Boolean)
              .join(' '),
          }}
        >
          <aside
            aria-label="部品の一覧"
            hidden={!panels.left}
            className="min-h-0 border-r border-line bg-surface"
          >
            <Palette
              units={paletteUnits}
              notice={notice}
              onAdd={add}
              onImport={() => setImporting(true)}
              refreshable={refreshable}
              onRefresh={(unit) => void refresh(unit.name.slice(SUBGRAPH_PREFIX.length))}
              onRemove={(unit) => {
                // 使っているノードがある部分モデルは消さない
                const name = unit.name.slice(SUBGRAPH_PREFIX.length)
                const users = subgraphUsers(whole, name)
                if (users.nodes.length > 0) {
                  notify(
                    `部分モデル sub.${name} はノード ${users.nodes.join(', ')} で使っているので消せません. 先にそのノードを消してください`,
                  )
                  return
                }
                if (users.subgraphs.length > 0) {
                  notify(
                    `部分モデル sub.${name} は部分モデル ${users.subgraphs.map((item) => `sub.${item}`).join(', ')} の中で使っているので消せません. 先にそちらを消してください`,
                  )
                  return
                }
                setHistory(removeSubgraph(whole, name))
                notify(`部分モデル sub.${name} を消しました`)
              }}
            />
          </aside>
          <main id="canvas" tabIndex={-1} className="relative min-h-0 outline-none">
            {/* 左右の欄を畳む (狭い画面でキャンバスを広く使う) */}
            <button
              type="button"
              onClick={() => setPanels((current) => ({ ...current, left: !current.left }))}
              aria-expanded={panels.left}
              aria-label={panels.left ? '部品の一覧を畳む' : '部品の一覧を開く'}
              title={panels.left ? '部品の一覧を畳む' : '部品の一覧を開く'}
              className="absolute top-2 left-2 z-10 flex size-7 items-center justify-center rounded-md border border-line bg-surface text-ink-2 shadow-sm hover:bg-surface-2"
            >
              {panels.left ? (
                <PanelLeftClose className="size-4" aria-hidden />
              ) : (
                <PanelLeftOpen className="size-4" aria-hidden />
              )}
            </button>
            <button
              type="button"
              onClick={() => setPanels((current) => ({ ...current, right: !current.right }))}
              aria-expanded={panels.right}
              aria-label={panels.right ? '詳細を畳む' : '詳細を開く'}
              title={panels.right ? '詳細を畳む' : '詳細を開く'}
              className="absolute top-2 right-2 z-10 flex size-7 items-center justify-center rounded-md border border-line bg-surface text-ink-2 shadow-sm hover:bg-surface-2"
            >
              {panels.right ? (
                <PanelRightClose className="size-4" aria-hidden />
              ) : (
                <PanelRightOpen className="size-4" aria-hidden />
              )}
            </button>
            {editingName && (
              <aside
                aria-label="編集している部分モデル"
                className="absolute top-2 left-1/2 z-10 flex max-w-[calc(100%-1rem)] -translate-x-1/2 flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-accent bg-surface px-3 py-1.5 text-xs shadow-sm"
              >
                <nav aria-label="部分モデルの階層" className="flex flex-wrap items-center gap-1">
                  <button
                    type="button"
                    data-edit-back
                    onClick={() => openPath([])}
                    className="rounded px-1 whitespace-nowrap text-accent-text hover:bg-surface-2"
                  >
                    本体
                  </button>
                  {validPath.map((name, index) => (
                    <span key={name} className="flex items-center gap-1">
                      <span aria-hidden className="text-ink-3">
                        ›
                      </span>
                      {index === validPath.length - 1 ? (
                        <span aria-current="page" className="font-mono">
                          sub.{name}
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => openPath(validPath.slice(0, index + 1))}
                          className="rounded px-1 font-mono whitespace-nowrap text-accent-text hover:bg-surface-2"
                        >
                          sub.{name}
                        </button>
                      )}
                    </span>
                  ))}
                </nav>
                <span
                  className="text-ink-3"
                  title="変更はこの定義の中の部分モデルに入り, それを使う全てのノードに反映されます"
                >
                  を編集中 (変更は使っているノード {subgraphUsers(whole, editingName).nodes.length}{' '}
                  個すべてに反映)
                </span>
                {usage && Object.keys(usage.shapes).length > 0 && (
                  <span
                    className="text-ink-3"
                    title="形の推定は, 本体でこの部分モデルを使う最初のノードに入る形で行います (保存する入力の形は変えません)"
                  >
                    形はノード {usage.node} に入る形で推定
                  </span>
                )}
                {wholeErrors > 0 && (
                  <span className="text-critical-text">本体に誤り {wholeErrors} 件</span>
                )}
                <span role="status" className="sr-only">
                  sub.{editingName} を編集中
                </span>
              </aside>
            )}
            <Canvas
              onDropUnit={(name, position) => {
                const unit = unitMap.get(name)
                if (unit?.available) add(unit, false, position)
              }}
              onOpenSubgraph={(name) => openPath([...validPath, name])}
              spec={spec}
              units={unitMap}
              check={check.data}
              selected={current}
              onChange={setSpec}
              onSelect={setSelected}
            />
          </main>
          <aside
            aria-label="詳細"
            hidden={!panels.right}
            className="min-h-0 overflow-y-auto border-l border-line bg-surface"
          >
            {check.error && (
              <div className="p-3">
                <ErrorBox error={check.error} live={false} />
              </div>
            )}
            <Inspector
              spec={spec}
              units={unitMap}
              check={check.data}
              selected={current}
              onChange={setSpec}
              onSelect={setSelected}
              onDuplicate={duplicate}
              onEditSubgraph={(name) => openPath([...validPath, name])}
              forkable={forkable}
              onForkSubgraph={(node) => {
                const forked = forkSubgraph(whole, node, editingName)
                if (!forked) return
                setHistory(forked.spec)
                notify(
                  `ノード ${node} は sub.${forked.name} (写したもの) を使うようにしました. 中身を変えても他のノードは変わりません`,
                )
              }}
            />
          </aside>
        </div>
      </div>
      {importing && (
        <SubgraphDialog
          current={ownFile}
          onImport={(name, source, origin) => {
            // 写すと同じ中身のものは 1 つにまとまるので, 変わらない場合は
            // 履歴に残さずに知らせるだけ
            // (部分モデルを編集している間も, 定義全体に写す)
            const embedded = embedSubgraph(whole, name, source, { origin })
            const unit = `${SUBGRAPH_PREFIX}${embedded.name}`
            if (
              whole.subgraphs?.[embedded.name] &&
              JSON.stringify(embedded.spec) === JSON.stringify(whole)
            ) {
              notify(`${unit} は既にあります (同じ中身)`)
              return
            }
            setHistory(embedded.spec)
            notify(
              embedded.name === name
                ? `${unit} を部品の一覧に加えました`
                : `${unit} として加えました (同じ名前で中身の違うものがあるか, 名前に使えないため)`,
            )
          }}
          onClose={() => setImporting(false)}
        />
      )}
    </>
  )
}
