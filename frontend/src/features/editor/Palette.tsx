import clsx from 'clsx'
import { ChevronDown, ChevronRight, Plus, Search } from 'lucide-react'
import { useId, useMemo, useState } from 'react'

import type { UnitSpec } from '../../lib/graph'

// 出どころの表示名と並び順
const SOURCES: Record<string, string> = {
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
  segmentation_models_pytorch: 'segmentation_models_pytorch',
  custom: '自作',
}

/** 部品を出どころと分類ごとに並べ, 検索して選べる一覧 */
export function Palette({ units, onAdd }: { units: UnitSpec[]; onAdd: (unit: UnitSpec) => void }) {
  const searchId = useId()
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState<Record<string, boolean>>({ ml_modelcontrol: true })

  // 名前と説明で絞り込み, 出どころごとにまとめる
  const groups = useMemo(() => {
    const text = query.trim().toLowerCase()
    const matched = units.filter(
      (unit) =>
        !text ||
        unit.name.toLowerCase().includes(text) ||
        unit.description.toLowerCase().includes(text),
    )
    const order = Object.keys(SOURCES)
    const bySource = new Map<string, UnitSpec[]>()
    for (const unit of matched) {
      bySource.set(unit.source, [...(bySource.get(unit.source) ?? []), unit])
    }
    const rank = (source: string) => {
      const index = order.indexOf(source)
      return index < 0 ? order.length : index
    }
    return [...bySource.entries()].sort(([a], [b]) => rank(a) - rank(b))
  }, [units, query])

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
          placeholder="部品を探す (例: conv, gcn)"
          className="h-8 w-full rounded-md border border-line bg-bg pr-2 pl-7 text-xs"
        />
      </div>
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
                {SOURCES[source] ?? source}
                <span className="ml-auto font-normal text-ink-3">{items.length}</span>
              </button>
              {expanded && (
                <ul>
                  {items.map((unit) => (
                    <li key={unit.name}>
                      <button
                        type="button"
                        disabled={!unit.available}
                        onClick={() => onAdd(unit)}
                        title={unit.available ? unit.description : unit.reason}
                        className={clsx(
                          'group flex w-full items-start gap-1.5 rounded px-2 py-1 text-left',
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
                            {unit.available ? unit.description || unit.category : unit.reason}
                          </span>
                        </span>
                      </button>
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
