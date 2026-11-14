import { useQuery } from '@tanstack/react-query'
import { ReactFlowProvider, useReactFlow } from '@xyflow/react'
import { useCallback, useEffect, useEffectEvent, useMemo, useState } from 'react'

import { api } from './api/client'
import { ErrorBox, Spinner } from './components/ui'
import { Canvas } from './features/editor/Canvas'
import { Inspector } from './features/editor/Inspector'
import { Palette } from './features/editor/Palette'
import { Toolbar } from './features/editor/Toolbar'
import { addNode, emptySpec, pasteNode, type NodeDef, type Spec, type UnitSpec } from './lib/graph'
import { useHistory } from './lib/history'
import { arrangeSpec, placeMissing } from './lib/layout'

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
  const [added, setAdded] = useState(0)
  const flow = useReactFlow()

  // 定義が落ち着いたら確認と形の推定 (前の結果は次が届くまで表示)
  const debounced = useDebounced(spec, 500)
  const check = useQuery({
    queryKey: ['check', debounced],
    queryFn: () => api.check(debounced),
    placeholderData: (previous) => previous,
    staleTime: Infinity,
  })

  const unitMap = useMemo(
    () => new Map((units.data ?? []).map((unit) => [unit.name, unit])),
    [units.data],
  )
  const dirty = useMemo(() => JSON.stringify(spec) !== saved, [spec, saved])

  // 新しく置くものは, キャンバスに見えている範囲の真ん中 (少しずつずらす)
  const center = () => {
    const box = document.querySelector('.react-flow')?.getBoundingClientRect()
    const point = box
      ? flow.screenToFlowPosition({ x: box.left + box.width / 2, y: box.top + box.height / 2 })
      : { x: 300, y: 120 }
    const shift = (added % 5) * 30
    setAdded((count) => count + 1)
    return { x: Math.round(point.x - 90 + shift), y: Math.round(point.y - 40 + shift) }
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
  const undoSpec = useCallback(() => {
    if (!previous) return
    if (selected !== null && !has(previous, selected)) setSelected(null)
    undo()
  }, [previous, selected, undo])
  const redoSpec = useCallback(() => {
    if (!following) return
    if (selected !== null && !has(following, selected)) setSelected(null)
    redo()
  }, [following, selected, redo])

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
    paste(node, { x: (node.ui?.x ?? 0) + 40, y: (node.ui?.y ?? 0) + 40 })
  }

  // 写す (Ctrl+C), 貼る (Ctrl+V), 複製する (Ctrl+D). 部品のノードを選んで
  // いる場合だけ (それ以外はブラウザの標準の操作)
  const onClipboardKey = useEffectEvent((key: string): boolean => {
    const node = current !== null ? spec.nodes[current] : undefined
    // 文字を選んでいる場合の Ctrl+C は, その文字の写し (ブラウザの標準)
    const text = !(window.getSelection()?.isCollapsed ?? true)
    if (key === 'c' && node && !text) setClipboard(node)
    else if (key === 'v' && clipboard) paste(clipboard, center())
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
  const add = (unit: UnitSpec) => {
    const next = addNode(spec, unit, center())
    const name = Object.keys(next.nodes).find((key) => !(key in spec.nodes)) ?? null
    setSpec(next)
    setSelected(name)
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
            const placed = placeMissing(loaded, unitMap)
            history.reset(placed)
            setSaved(JSON.stringify(placed))
            setFileName(name)
            setSelected(null)
          }}
          ownFile={ownFile}
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
            setSpec((current) => arrangeSpec(current, unitMap))
            // 並べ直した全体が見えるように表示の範囲を合わせる
            window.requestAnimationFrame(() => void flow.fitView({ duration: 200 }))
          }}
          onUndo={previous ? undoSpec : undefined}
          onRedo={following ? redoSpec : undefined}
        />
        <div className="grid min-h-0 flex-1 grid-cols-[16rem_minmax(0,1fr)_20rem]">
          <aside aria-label="部品の一覧" className="min-h-0 border-r border-line bg-surface">
            <Palette units={units.data} onAdd={add} />
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
    </>
  )
}
