import clsx from 'clsx'
import { ChevronDown, ChevronRight, Layers, Plus, RefreshCw, Search, Trash2 } from 'lucide-react'
import { useDeferredValue, useId, useMemo, useState } from 'react'

import type { UnitSpec } from '../../lib/graph'
import { firstMatch, matchesQuery } from '../../lib/search'
import { UNIT_DRAG_TYPE } from '../../lib/flow'

// 出どころの表示名と並び順
const SOURCES: Record<string, string> = {
  subgraph: '部分モデル',
  ml_modelcontrol: '演算',
  'torch.nn': 'PyTorch',
  torch_geometric: 'PyTorch Geometric',
  transformers: 'Hugging Face',
  torchvision: 'torchvision',
  timm: 'timm',
  torchaudio: 'torchaudio',
  einops: 'einops',
  kornia: 'Kornia',
  monai: 'MONAI',
  segmentation_models_pytorch: 'smp (segmentation_models_pytorch)',
  custom: '自作',
}

/** 部品を出どころと分類ごとに並べ, 検索して選べる一覧 */
export function Palette({
  units,
  onAdd,
  onImport,
  onRemove,
  notice,
  refreshable,
  onRefresh,
}: {
  units: UnitSpec[]
  /** 部分モデルの操作の結果の知らせ (変わるたびに検索を消して見せる) */
  notice?: { text: string } | null
  /** 部品を置く (focus はキーボードで置いた場合) */
  onAdd: (unit: UnitSpec, focus?: boolean) => void
  /** 保存した定義を部分モデルとして写す対話を開く */
  onImport?: () => void
  /** 部分モデルを定義から消す */
  onRemove?: (unit: UnitSpec) => void
  /** 写した元の定義から写し直せる部分モデルの部品の名前 */
  refreshable?: Set<string>
  /** 部分モデルを写した元の定義の今の中身で写し直す */
  onRefresh?: (unit: UnitSpec) => void
}) {
  const searchId = useId()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState<Record<string, boolean>>({
    subgraph: true,
    ml_modelcontrol: true,
  })
  // 部分モデルを加えたら, 検索で隠れないように検索を消す (知らせが変わった
  // 描画の中で前の知らせと比べる)
  const [seen, setSeen] = useState(notice)
  if (notice !== seen) {
    setSeen(notice)
    if (notice) setQuery('')
  }

  // 名前と説明で絞り込み, 出どころごとにまとめる
  // 打鍵の間は前の結果を見せ, 多くの部品の描き直しで入力を止めない
  const deferred = useDeferredValue(query)
  const groups = useMemo(() => {
    const text = deferred.trim().toLowerCase()
    const matched = units.filter((unit) => matchesQuery(unit, text))
    const order = Object.keys(SOURCES)
    const bySource = new Map<string, UnitSpec[]>()
    for (const unit of matched) {
      const list = bySource.get(unit.source)
      if (list) list.push(unit)
      else bySource.set(unit.source, [unit])
    }
    const rank = (source: string) => {
      const index = order.indexOf(source)
      return index < 0 ? order.length : index
    }
    return [...bySource.entries()].sort(([a], [b]) => rank(a) - rank(b))
  }, [units, deferred])

  return (
    <div className="flex h-full flex-col">
      <label htmlFor={searchId} className="sr-only">
        部品を探す
      </label>
      <div className="relative border-b border-line p-2">
        <Search className="absolute top-4 left-4 size-3.5 text-ink-3" aria-hidden />
        <input
          id={searchId}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          // Enter で, 絞り込んだ一番上の使える部品を置く (キーボードだけで置くため)
          // (日本語の変換の確定の Enter では置かない. Safari は keyCode 229)
          onKeyDown={(event) => {
            if (event.key !== 'Enter' || event.nativeEvent.isComposing) return
            if (event.keyCode === 229 || event.repeat) return
            const first = firstMatch(units, query)
            if (!first) return
            onAdd(first, true)
            // 置いた後は検索を消す (次に探すときに前の文字が残らないように)
            setQuery('')
          }}
          aria-describedby={`${searchId}-hint`}
          placeholder="部品を探す (例: conv, gcn)"
          className="h-8 w-full rounded-md border border-line bg-bg pr-2 pl-7 text-xs"
        />
        <p id={`${searchId}-hint`} className="sr-only">
          Enter で一番合う部品 (名前が一致するもの) を置きます
        </p>
      </div>
      {onImport && (
        <div className="border-b border-line p-2">
          <button
            type="button"
            onClick={onImport}
            title="保存した定義か例を, この定義の中で 1 つの部品 (部分モデル) として使えるようにする"
            className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-left text-xs text-ink-2 hover:bg-surface-2"
          >
            <Layers className="size-3.5 text-ink-3" aria-hidden />
            別の定義を部品にする
          </button>
          <p role="status" className="px-2 pt-1 text-[10px] text-ink-2 empty:hidden">
            {notice?.text}
          </p>
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-y-auto p-1">
        {groups.map(([source, items]) => {
          const expanded = open[source] || query.trim() !== ''
          return (
            <section key={source} className="mb-1">
              <button
                type="button"
                onClick={() => setOpen((current) => ({ ...current, [source]: !expanded }))}
                aria-expanded={expanded}
                className="flex w-full items-center gap-1 rounded px-2 py-1 text-left text-xs font-semibold text-ink-2 hover:bg-surface-2"
              >
                {expanded ? (
                  <ChevronDown className="size-3.5" aria-hidden />
                ) : (
                  <ChevronRight className="size-3.5" aria-hidden />
                )}
                <span className="min-w-0 truncate" title={SOURCES[source] ?? source}>
                  {SOURCES[source] ?? source}
                </span>
                <span className="ml-auto shrink-0 font-normal text-ink-3">{items.length}</span>
              </button>
              {expanded && (
                <ul>
                  {items.map((unit) => (
                    <li key={unit.name} className="flex items-start">
                      <button
                        type="button"
                        disabled={!unit.available}
                        onClick={() => onAdd(unit)}
                        // キャンバスにドラッグして, 落とした位置に置ける
                        draggable={unit.available}
                        onDragStart={(event) => {
                          event.dataTransfer.setData(UNIT_DRAG_TYPE, unit.name)
                          event.dataTransfer.effectAllowed = 'copy'
                        }}
                        title={unit.available ? unitTitle(unit) : appReason(unit.reason)}
                        className={clsx(
                          'group flex min-w-0 flex-1 items-start gap-1.5 rounded px-2 py-1 text-left',
                          unit.available ? 'hover:bg-surface-2' : 'cursor-not-allowed opacity-50',
                        )}
                      >
                        <Plus
                          className="mt-0.5 size-3 shrink-0 text-ink-3 group-hover:text-accent-text"
                          aria-hidden
                        />
                        <span className="min-w-0">
                          <span className="block truncate font-mono text-xs">{unit.name}</span>
                          <span className="block truncate text-[10px] text-ink-3">
                            {unit.available
                              ? unit.description || unit.category
                              : appReason(unit.reason)}
                          </span>
                        </span>
                      </button>
                      {unit.source === 'subgraph' && onRefresh && refreshable?.has(unit.name) && (
                        <button
                          type="button"
                          onClick={() => onRefresh(unit)}
                          aria-label={`部分モデル ${unit.name} を元の定義から写し直す`}
                          title={`部分モデル ${unit.name} を, 写した元の定義の今の中身で写し直す`}
                          className="flex size-6 shrink-0 items-center justify-center rounded text-ink-3 hover:bg-surface-2 hover:text-accent-text"
                        >
                          <RefreshCw className="size-3" aria-hidden />
                        </button>
                      )}
                      {unit.source === 'subgraph' && onRemove && (
                        <button
                          type="button"
                          onClick={() => onRemove(unit)}
                          aria-label={`部分モデル ${unit.name} を消す`}
                          title={`部分モデル ${unit.name} を定義から消す`}
                          className="flex size-6 shrink-0 items-center justify-center rounded text-ink-3 hover:bg-surface-2 hover:text-critical-text"
                        >
                          <Trash2 className="size-3" aria-hidden />
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )
        })}
        {groups.length === 0 && (
          <p className="p-3 text-xs text-ink-3">見つかりません. 別の言葉で探してください.</p>
        )}
      </div>
    </div>
  )
}

/** 使えない理由の入れ方を, このアプリの入れ方 (make setup EXTRAS=...) にする */
function appReason(reason: string): string {
  return reason.replace(/pip install -e "\.\[(\w+)\]"/g, 'make setup EXTRAS=$1')
}

/** 部品の説明 (学習済みの重みを読める部品は, ライセンスを詳細で確かめる案内を足す) */
function unitTitle(unit: UnitSpec): string {
  if (!unit.weights_params?.length) return unit.description
  return `${unit.description} (学習済みの重みを読める. ライセンスは置いた後の詳細で確かめる)`
}
