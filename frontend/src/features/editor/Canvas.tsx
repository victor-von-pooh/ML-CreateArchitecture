import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from '@xyflow/react'
import { useMemo, useRef, useState } from 'react'

import type { CheckResult } from '../../api/client'
import { flowEdges, flowNodes, moveItem, UNIT_DRAG_TYPE } from '../../lib/flow'
import { useTheme } from '../../lib/themeContext'
import {
  connect,
  disconnect,
  parseRef,
  removeItem,
  SUBGRAPH_PREFIX,
  type Spec,
  type UnitSpec,
} from '../../lib/graph'
import { outsideStage } from '../../lib/stage'
import { InputNode, OutputNode, UnitNode } from './nodes'

// キャンバスのノードの種類と表示
const nodeTypes = { unit: UnitNode, modelInput: InputNode, modelOutput: OutputNode }

/** モデルの定義を編集するキャンバス */
// 最初に表示の範囲を合わせるときの設定 (描くたびに作らない定数)
const FIT_OPTIONS = { maxZoom: 1, padding: 0.1 }

export function Canvas({
  spec,
  units,
  check,
  selected,
  onChange,
  onSelect,
  onOpenSubgraph,
  onDropUnit,
}: {
  spec: Spec
  units: Map<string, UnitSpec>
  check: CheckResult | undefined
  selected: string | null
  /**
   * 定義の変更 (merge に同じ印の続けての変更は, 元に戻すで 1 回分). 同じ処理の
   * 中で続けて呼ばれる場合 (線とノードを一緒に消すなど) のために, 今の定義から
   * 新しい定義を作る関数も受け取る
   */
  onChange: (spec: Spec | ((current: Spec) => Spec), merge?: string) => void
  onSelect: (id: string | null) => void
  /** 部分モデルのノードをダブルクリックして, 中身の編集を始める */
  onOpenSubgraph?: (name: string) => void
  /** 部品の一覧から落とした部品を, キャンバスの座標の位置に置く */
  onDropUnit?: (unit: string, position: { x: number; y: number }) => void
}) {
  // 計測したノードの大きさ (線の位置の計算に使うので, 作り直しても保持)
  const flow = useReactFlow()
  const [sizes, setSizes] = useState<Record<string, { width: number; height: number }>>({})
  const nodes = useMemo(
    () =>
      flowNodes(spec, units, check, selected).map((node) =>
        sizes[node.id] ? { ...node, measured: sizes[node.id] } : node,
      ),
    [spec, units, check, selected, sizes],
  )
  // 選んだ線 (Delete で消せるように, 定義とは別に保持)
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null)
  // ノードを選んだら (貼り付けなど), 線の選択は外れたものとする (続けての
  // Delete で線まで消さないため)
  const activeEdge = selected === null ? selectedEdge : null
  const edges = useMemo(() => flowEdges(spec, units, activeEdge), [spec, units, activeEdge])
  const { theme } = useTheme()

  // Delete キーで消すと, キャンバスは線とノードの削除を別々に知らせるので,
  // 同じ操作の中の削除を元に戻す 1 回分にまとめる印
  const deleting = useRef<string | null>(null)
  const deleteMark = () => {
    if (deleting.current === null) {
      deleting.current = `delete:${Date.now()}`
      window.setTimeout(() => {
        deleting.current = null
      }, 0)
    }
    return deleting.current
  }

  // ノードの移動, 選択, 削除を定義に反映
  const onNodesChange = (changes: NodeChange[]) => {
    let next = spec
    const measured: Record<string, { width: number; height: number }> = {}
    const moved: string[] = []
    for (const change of changes) {
      if (change.type === 'dimensions' && change.dimensions) {
        measured[change.id] = change.dimensions
      } else if (change.type === 'position' && change.position) {
        next = moveItem(next, change.id, change.position)
        moved.push(change.id)
      } else if (change.type === 'select' && change.selected) {
        onSelect(change.id)
      } else if (change.type === 'remove') {
        if (selected === change.id) onSelect(null)
      }
    }
    if (Object.keys(measured).length > 0) setSizes((current) => ({ ...current, ...measured }))
    // 同じノードのドラッグの移動は 1 回分にまとめる
    // 消すのは今の定義から (同じ処理の中で先に線を消した結果を上書きしない)
    const removedIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []))
    if (removedIds.length > 0) {
      onChange(
        (current) => removedIds.reduce((value, id) => removeItem(value, units, id), current),
        deleteMark(),
      )
      // 消した後は focus をキャンバスに置く (ページの先頭に戻らないように)
      window.requestAnimationFrame(() => document.getElementById('canvas')?.focus())
    }
    if (next !== spec) onChange(next, moved.length > 0 ? `move:${moved.join(',')}` : undefined)
  }

  // 線の削除を定義に反映
  const onEdgesChange = (changes: EdgeChange[]) => {
    for (const change of changes) {
      if (change.type === 'select') {
        setSelectedEdge(change.selected ? change.id : null)
        if (change.selected) onSelect(null)
      }
    }
    const removedIds = changes.flatMap((change) => (change.type === 'remove' ? [change.id] : []))
    if (removedIds.length > 0) {
      onChange(
        (current) => removedIds.reduce((value, id) => disconnect(value, units, id), current),
        deleteMark(),
      )
    }
  }

  // 線を引いたら, つなぎ先を定義に追加
  const onConnect = (connection: Connection) => {
    if (!connection.sourceHandle || !connection.targetHandle) return
    onChange(
      connect(spec, units, {
        source: connection.source,
        sourcePort: connection.sourceHandle.replace(/^out:/, ''),
        target: connection.target,
        targetPort: connection.targetHandle.replace(/^in:/, ''),
      }),
    )
  }

  // 自分自身へのつなぎ方は不可
  const isValidConnection = (connection: Connection | Edge) =>
    connection.source !== connection.target &&
    parseRef(connection.source).name !== connection.target

  return (
    <ReactFlow
      // 部品の一覧からのドラッグを受け, 落とした位置に置く
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes(UNIT_DRAG_TYPE)) {
          event.preventDefault()
          event.dataTransfer.dropEffect = 'copy'
        }
      }}
      onDrop={(event) => {
        const unit = event.dataTransfer.getData(UNIT_DRAG_TYPE)
        if (!unit || !onDropUnit) return
        event.preventDefault()
        const point = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY })
        onDropUnit(unit, { x: Math.round(point.x - 100), y: Math.round(point.y - 30) })
      }}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeDoubleClick={(_, node) => {
        const unit = spec.nodes[node.id]?.unit
        if (unit?.startsWith(SUBGRAPH_PREFIX)) onOpenSubgraph?.(unit.slice(SUBGRAPH_PREFIX.length))
      }}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      isValidConnection={isValidConnection}
      onPaneClick={() => {
        onSelect(null)
        setSelectedEdge(null)
      }}
      deleteKeyCode={['Backspace', 'Delete']}
      // React Flow は Delete を document で聞くので, focus が部品の範囲の外 (殻の上の帯など)
      // にあるときは消さない
      onBeforeDelete={() => Promise.resolve(!outsideStage(document.activeElement))}
      fitView
      fitViewOptions={FIT_OPTIONS}
      // 長いモデルの全体も見られるように, 小さくする限度を下げる
      minZoom={0.2}
      colorMode={theme}
      proOptions={{ hideAttribution: true }}
      aria-label="モデルのキャンバス"
    >
      <Background gap={20} />
      <Controls showInteractive={false} />
      {/* 狭い画面ではノードを隠すので出さない */}
      <MiniMap pannable zoomable className="max-2xl:!hidden" />
    </ReactFlow>
  )
}
