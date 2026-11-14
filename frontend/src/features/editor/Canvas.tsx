import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from '@xyflow/react'
import { useMemo, useState } from 'react'

import type { CheckResult } from '../../api/client'
import { flowEdges, flowNodes, moveItem } from '../../lib/flow'
import { useTheme } from '../../lib/themeContext'
import {
  connect,
  disconnect,
  parseRef,
  removeItem,
  type Spec,
  type UnitSpec,
} from '../../lib/graph'
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
}: {
  spec: Spec
  units: Map<string, UnitSpec>
  check: CheckResult | undefined
  selected: string | null
  /** 定義の変更 (merge に同じ印の続けての変更は, 元に戻すで 1 回分) */
  onChange: (spec: Spec, merge?: string) => void
  onSelect: (id: string | null) => void
}) {
  // 計測したノードの大きさ (線の位置の計算に使うので, 作り直しても保持)
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
        next = removeItem(next, units, change.id)
        if (selected === change.id) onSelect(null)
      }
    }
    if (Object.keys(measured).length > 0) setSizes((current) => ({ ...current, ...measured }))
    // 同じノードのドラッグの移動は 1 回分にまとめる
    if (next !== spec) onChange(next, moved.length > 0 ? `move:${moved.join(',')}` : undefined)
  }

  // 線の削除を定義に反映
  const onEdgesChange = (changes: EdgeChange[]) => {
    let next = spec
    for (const change of changes) {
      if (change.type === 'remove') next = disconnect(next, units, change.id)
      else if (change.type === 'select') {
        setSelectedEdge(change.selected ? change.id : null)
        if (change.selected) onSelect(null)
      }
    }
    if (next !== spec) onChange(next)
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
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      isValidConnection={isValidConnection}
      onPaneClick={() => {
        onSelect(null)
        setSelectedEdge(null)
      }}
      deleteKeyCode={['Backspace', 'Delete']}
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
