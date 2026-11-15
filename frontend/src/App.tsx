import { useQuery } from '@tanstack/react-query'
import { ReactFlowProvider, useReactFlow } from '@xyflow/react'
import { useCallback, useEffect, useEffectEvent, useMemo, useState } from 'react'

import { api } from './api/client'
import { ErrorBox, Spinner } from './components/ui'
import { Canvas } from './features/editor/Canvas'
import { Inspector } from './features/editor/Inspector'
import { Palette } from './features/editor/Palette'
import { Toolbar } from './features/editor/Toolbar'
import { SubgraphDialog } from './features/editor/SubgraphDialog'
import {
  addNode,
  embedSubgraph,
  emptySpec,
  pasteNode,
  removeSubgraph,
  SUBGRAPH_PREFIX,
  subgraphUnits,
  subgraphUsers,
  type NodeDef,
  type Spec,
  type UnitSpec,
} from './lib/graph'
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
  const spec = history.value
  const setSpec = history.set
  const [fileName, setFileName] = useState('model')
  // 今の定義を開いた (か保存した) 保存した定義の名前 (上書きの確認に使う)
  const [ownFile, setOwnFile] = useState<string | null>(null)
  // 保存した (か開いた) 時点の定義. 今の定義と比べて, 保存していない変更を判定
  const [saved, setSaved] = useState(() => JSON.stringify(emptySpec()))
  const [selected, setSelected] = useState<string | null>(null)
  const flow = useReactFlow()

  // 定義が落ち着いたら確認と形の推定 (前の結果は次が届くまで表示)
  const debounced = useDebounced(spec, 500)
  const check = useQuery({
    queryKey: ['check', debounced],
    queryFn: () => api.check(debounced),
    placeholderData: (previous) => previous,
    staleTime: Infinity,
  })

  // 登録簿の部品に, 定義ごとの部分モデルの部品 (sub.<名前>) を加えたもの
  const baseUnits = useMemo(
    () => new Map((units.data ?? []).map((unit) => [unit.name, unit])),
    [units.data],
  )
  const unitsFor = useCallback(
    (target: Spec) => {
      const local = subgraphUnits(target)
      if (local.length === 0) return baseUnits
      return new Map([...baseUnits, ...local.map((unit) => [unit.name, unit] as const)])
    },
    [baseUnits],
  )
  const unitMap = useMemo(() => unitsFor(spec), [unitsFor, spec])
  const paletteUnits = useMemo(
    () => [...subgraphUnits(spec), ...(units.data ?? [])],
    [spec, units.data],
  )
  const [importing, setImporting] = useState(false)
  const dirty = useMemo(() => JSON.stringify(spec) !== saved, [spec, saved])

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
    (target: Spec) => {
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
    [selected, spec],
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
        if (document.activeElement instanceof HTMLElement) document.activeElement.blur()
        window.requestAnimationFrame(() =>
          document.querySelector<HTMLButtonElement>('[data-save]:not(:disabled)')?.click(),
        )
        return
      }
      if (target?.closest('dialog[open], textarea, [contenteditable="true"]')) return
      if (
        target instanceof HTMLInputElement &&
        !['checkbox', 'radio', 'button', 'submit', 'range', 'color'].includes(target.type)
      ) {
        return
      }
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return
      const key = event.key.toLowerCase()
      if (key === 'z' && !event.shiftKey) undoSpec()
      else if ((key === 'z' && event.shiftKey) || (key === 'y' && event.ctrlKey)) redoSpec()
      else if (event.shiftKey || !['c', 'v', 'd'].includes(key)) return
      // 押し続けた場合の繰り返しでは, 写しを増やさない
      else if (event.repeat) return
      else if (!onClipboardKey(key)) return
      event.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [undoSpec, redoSpec])

  // 部品はキャンバスの真ん中に置いて選択
  // 部品のノードを選んでいる場合は, その右 (つなぐ先になることが多い) に置く
  // focus: キーボードで置いた場合 (検索欄の Enter) は詳細の最初の欄に移る
  const add = (unit: UnitSpec, focus = false) => {
    const from = current !== null ? (spec.nodes[current] ?? spec.inputs[current]) : undefined
    const near =
      from?.ui?.x !== undefined && from.ui.y !== undefined
        ? { x: from.ui.x + 280, y: from.ui.y }
        : undefined
    const next = addNode(spec, unit, center(unitSize(unit), near))
    const name = Object.keys(next.nodes).find((key) => !(key in spec.nodes)) ?? null
    setSpec(next)
    setSelected(name)
    // キーボードで続けて操作できるように, 詳細の最初の欄に移る
    if (!focus) return
    window.requestAnimationFrame(() =>
      document
        .querySelector<HTMLElement>(
          'aside[aria-label="詳細"] select, aside[aria-label="詳細"] input',
        )
        ?.focus(),
    )
  }

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
        <Toolbar
          // 開いた定義が変わったら, 保存する名前の欄を作り直す
          key={fileName}
          spec={spec}
          fileName={fileName}
          dirty={dirty}
          check={check.data}
          checking={check.isFetching || debounced !== spec}
          onChange={setSpec}
          onLoad={(name, loaded, source) => {
            setOwnFile(source === 'model' ? name : null)
            // 位置の無いもの (手で書いた定義など) は, 開いたときに一度だけ
            // 段に分けた位置を書く (後の編集で動かないように)
            const placed = placeMissing(loaded, unitsFor(loaded))
            history.reset(placed)
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
              () => void flow.fitView({ duration: 200, maxZoom: 1, minZoom: 0.5, padding: 0.1 }),
            )
          }}
          onUndo={previous ? undoSpec : undefined}
          onRedo={following ? redoSpec : undefined}
        />
        <div className="grid min-h-0 flex-1 grid-cols-[16rem_minmax(0,1fr)_20rem]">
          <aside aria-label="部品の一覧" className="min-h-0 border-r border-line bg-surface">
            <Palette
              units={paletteUnits}
              onAdd={add}
              onImport={() => setImporting(true)}
              onRemove={(unit) => {
                // 使っているノードがある部分モデルは消さない
                const name = unit.name.slice(SUBGRAPH_PREFIX.length)
                const users = subgraphUsers(spec, name)
                if (users.length > 0) {
                  window.alert(
                    `部分モデル ${name} は ${users.join(', ')} で使っているので消せません. 先にそのノードを消してください.`,
                  )
                  return
                }
                setSpec(removeSubgraph(spec, name))
              }}
            />
          </aside>
          <main className="min-h-0">
            <Canvas
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
            />
          </aside>
        </div>
      </div>
      {importing && (
        <SubgraphDialog
          current={ownFile}
          onImport={(name, source) => setSpec(embedSubgraph(spec, name, source).spec)}
          onClose={() => setImporting(false)}
        />
      )}
    </>
  )
}
