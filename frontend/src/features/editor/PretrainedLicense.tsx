import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { AlertTriangle, CheckCircle2, ExternalLink, Loader2 } from 'lucide-react'

import { api, type PretrainedInfo } from '../../api/client'
import { useDebounced } from '../../lib/debounce'
import { formatBytes, licenseParams } from '../../lib/license'

// 分類ごとの見出しと色 (商用で使えないものは誤り, 分からないものは注意の色)
const STATUS_STYLE: Record<string, { label: string; className: string }> = {
  commercial: { label: '商用で使える', className: 'border-accent/30 bg-accent/5' },
  noncommercial: {
    label: '警告: 商用で使えない',
    className: 'border-critical/40 bg-critical/5 text-critical-text',
  },
  unknown: {
    label: '警告: 商用で使えるか分からない',
    className: 'border-warning/50 bg-warning/10 text-warning-text',
  },
}

/** 学習済みの重みのライセンス (配布元から読んだもの) と警告. 重みを読む部品のノードの詳細に出す
 * (ノードを選び直したら前のノードの結果を出さないように, 呼ぶ側はノードの名前を key にする) */
export function PretrainedLicense({
  unit,
  params,
  weightsParams,
}: {
  unit: string
  params: Record<string, unknown>
  weightsParams: string[]
}) {
  // 打っている途中のモデルの名前で読みに行かないように, 落ち着いてから聞く
  const current = licenseParams(params, weightsParams)
  const target = useDebounced(current, 600)
  // 落ち着くまでの間は, 前の引数の結果を今の引数の結果として見せない
  const settling = current !== target
  const query = useQuery({
    queryKey: ['pretrained', unit, target],
    queryFn: () =>
      api.pretrained(unit, Object.fromEntries(JSON.parse(target) as [string, unknown][])),
    // バックエンドが結果を覚えるので, 画面では選び直しても聞き直さない
    staleTime: 10 * 60 * 1000,
    retry: false,
  })
  return (
    <section aria-label="学習済みの重みのライセンス" className="space-y-1 text-xs">
      <h3 className="font-semibold">学習済みの重みのライセンス</h3>
      {settling || query.isPending ? (
        <p role="status" className="flex items-center gap-1.5 text-ink-3">
          <Loader2 className="size-3.5 animate-spin" aria-hidden />
          配布元からライセンスを読んでいます
        </p>
      ) : query.isError ? (
        <p role="alert" className="text-critical-text">
          ライセンスを読めませんでした: {query.error.message}
        </p>
      ) : query.data ? (
        <LicenseCard info={query.data} />
      ) : (
        <p className="text-ink-3">今の引数では学習済みの重みを読みません.</p>
      )}
    </section>
  )
}

/** ライセンスの分類, 説明, どこから読んだか, 推論の一式での扱い, 表示の義務, 出どころの表記, 重みの大きさ
 * (値の無いものは出さない) */
export function LicenseCard({ info }: { info: PretrainedInfo }) {
  const style = STATUS_STYLE[info.status] ?? STATUS_STYLE.unknown!
  // 読めない大きさ (負の値など) は行ごと出さない
  const size = info.size_bytes == null ? null : formatBytes(info.size_bytes)
  return (
    <div
      // 結果は控えめに読み上げる (ノードを選ぶたびに割り込まないように)
      role="status"
      className={clsx('space-y-1 rounded-md border p-2', style.className)}
    >
      <p className="flex items-start gap-1.5 font-semibold">
        {info.alert ? (
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
        ) : (
          <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-accent-text" aria-hidden />
        )}
        {style.label}
      </p>
      <p className="[overflow-wrap:anywhere]">{info.message}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-ink-2">
        <dt className="text-ink-3">どこから</dt>
        <dd className="[overflow-wrap:anywhere]">{info.source}</dd>
        <dt className="text-ink-3">モデル</dt>
        <dd className="font-mono [overflow-wrap:anywhere]">{info.hub_id ?? info.model}</dd>
        {info.pinned_revision && (
          <>
            <dt className="text-ink-3">読んだ版</dt>
            <dd className="font-mono" title={info.pinned_revision}>
              {info.pinned_revision.slice(0, 12)}
            </dd>
          </>
        )}
        {size && (
          <>
            <dt className="text-ink-3">重みの大きさ</dt>
            <dd title={`${info.size_bytes} バイト`}>{size}</dd>
          </>
        )}
        <dt className="text-ink-3">推論の一式</dt>
        <dd>
          {info.redistributable
            ? '重みを写して固定できる'
            : '重みを写さず, 出どころの名前と版から読む'}
        </dd>
        {info.attribution && (
          <>
            <dt className="text-ink-3">表示の義務</dt>
            <dd className="[overflow-wrap:anywhere]">{info.attribution}</dd>
          </>
        )}
        {info.credit && (
          <>
            <dt className="text-ink-3">出どころの表記</dt>
            <dd className="select-all font-mono [overflow-wrap:anywhere]">{info.credit}</dd>
          </>
        )}
      </dl>
      {info.url && (
        <a
          href={info.url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-accent-text underline"
        >
          配布元のページ
          <ExternalLink className="size-3" aria-hidden />
          <span className="sr-only">(新しいタブで開く)</span>
        </a>
      )}
    </div>
  )
}
