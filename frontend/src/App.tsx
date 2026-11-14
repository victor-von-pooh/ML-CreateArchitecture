import { useQuery } from '@tanstack/react-query'
import { ReactFlowProvider } from '@xyflow/react'
import { useEffect, useMemo, useState } from 'react'

import { api } from './api/client'
import { ErrorBox, Spinner } from './components/ui'
import { Canvas } from './features/editor/Canvas'
import { Inspector } from './features/editor/Inspector'
import { Palette } from './features/editor/Palette'
import { Toolbar } from './features/editor/Toolbar'
import { addNode, emptySpec, type Spec, type UnitSpec } from './lib/graph'

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
  const units = useQuery({ queryKey: ['units'], queryFn: api.listUnits, staleTime: Infinity })
  const [spec, setSpec] = useState<Spec>(() => emptySpec())
  const [fileName, setFileName] = useState('model')
  const [saved, setSaved] = useState<Spec | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [added, setAdded] = useState(0)

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
  const dirty = saved !== spec && Object.keys(spec.nodes).length > 0

  // 保存していない変更がある場合は, 閉じる前に確認
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault()
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  // 部品を置く位置は, 少しずつずらして重ならないようにする
  const add = (unit: UnitSpec) => {
    const position = { x: 300 + (added % 5) * 40, y: 120 + (added % 5) * 40 }
    const next = addNode(spec, unit, position)
    const name = Object.keys(next.nodes).find((key) => !(key in spec.nodes)) ?? null
    setSpec(next)
    setSelected(name)
    setAdded((count) => count + 1)
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
    <ReactFlowProvider>
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
          onLoad={(name, loaded) => {
            setSpec(loaded)
            setSaved(loaded)
            setFileName(name)
            setSelected(null)
          }}
          onSaved={(name) => {
            setFileName(name)
            setSaved(spec)
          }}
          onSelect={setSelected}
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
              selected={selected}
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
              selected={selected}
              onChange={setSpec}
              onSelect={setSelected}
            />
          </aside>
        </div>
      </div>
    </ReactFlowProvider>
  )
}
