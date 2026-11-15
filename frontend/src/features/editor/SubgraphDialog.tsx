import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '../../api/client'
import { Dialog } from '../../components/Dialog'
import { ErrorBox, Spinner } from '../../components/ui'
import type { Spec, SubgraphOrigin } from '../../lib/graph'

/**
 * 保存した定義か同梱した例を選び, 今の定義の部分モデル (部品) として写す
 *
 * 写した部分モデルは部品の一覧の「部分モデル」に出て, 普通の部品と同じく
 * 置いてつなげる. 元の定義を後で変えても, 写したものは変わらない
 */
export function SubgraphDialog({
  current,
  onImport,
  onClose,
}: {
  /** 今開いている定義の保存した名前 (自分自身は写せないので一覧から除く) */
  current: string | null
  /** 選んだ定義を写す (写せない場合は誤りを投げる) */
  onImport: (name: string, spec: Spec, origin: SubgraphOrigin) => void
  onClose: () => void
}) {
  const models = useQuery({ queryKey: ['models'], queryFn: api.listModels })
  const examples = useQuery({
    queryKey: ['examples'],
    queryFn: api.listExamples,
    staleTime: Infinity,
  })
  // 定義を読み込み, 誤りがあるかを確かめる (形の推定はしない)
  const load = useMutation({
    mutationFn: async ({ source, name }: { source: 'model' | 'example'; name: string }) => {
      const model = source === 'model' ? await api.getModel(name) : await api.getExample(name)
      const checked = await api.check(model.spec as unknown as Spec, false)
      const errors = checked.issues.filter((issue) => issue.severity === 'error')
      return { model, errors }
    },
  })
  // 読み込んだ定義は, ダイアログを開いている間だけ写す (読み込みの途中で
  // 閉じた場合は写さない. 呼び出しごとの onSuccess は閉じた後は呼ばれない)
  const [importError, setImportError] = useState<unknown>(null)
  const pick = (source: 'model' | 'example', name: string) => {
    setImportError(null)
    load.mutate(
      { source, name },
      {
        onSuccess: ({ model, errors }) => {
          // 誤りのある定義は, 写すと使うノードも誤りになるので確かめる
          if (
            errors.length > 0 &&
            !window.confirm(
              `${model.name} には誤りが ${errors.length} 件あります (${errors[0]!.message}). ` +
                '写すと, 置いたノードは誤りになります (写した後に中身を直せます). 写しますか?',
            )
          ) {
            return
          }
          try {
            onImport(model.name, model.spec as unknown as Spec, { kind: source, name })
            onClose()
          } catch (error) {
            setImportError(error)
          }
        },
      },
    )
  }
  const own = (models.data ?? []).filter((model) => model.name !== current)
  return (
    <Dialog title="別の定義を部品にする" onClose={onClose}>
      <p className="mb-3 text-xs text-ink-2">
        選んだ定義を, 今の定義の中に部分モデルとして写します. 部品の一覧の「部分モデル」から置いて,
        普通の部品と同じようにつなげます (定義の入力が入力の口, 出力が出力の口).
        元の定義を後で変えても, 写したものは変わりません.
        今開いている定義自身は写せないので一覧に出しません.
      </p>
      {load.isPending && <Spinner label="定義を読み込み中" />}
      {load.error && <ErrorBox error={load.error} />}
      {importError ? <ErrorBox error={importError} /> : null}
      <h3 className="mb-1 text-xs font-semibold text-ink-2">保存した定義</h3>
      {models.isPending ? (
        <Spinner />
      ) : models.error ? (
        <ErrorBox error={models.error} />
      ) : own.length === 0 ? (
        <p className="mb-3 text-sm text-ink-3">写せる保存した定義はまだありません.</p>
      ) : (
        <ul className="mb-4 divide-y divide-line">
          {own.map((model) => (
            <li key={model.name}>
              <button
                type="button"
                disabled={load.isPending}
                onClick={() => pick('model', model.name)}
                className="flex w-full items-center justify-between px-2 py-2 text-left hover:bg-surface-2"
              >
                <span className="font-mono text-sm">{model.name}</span>
                <span className="text-xs text-ink-3">{model.updated_at.replace('T', ' ')}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <h3 className="mb-1 text-xs font-semibold text-ink-2">同梱した例</h3>
      {examples.isPending ? (
        <Spinner />
      ) : examples.error ? (
        <ErrorBox error={examples.error} />
      ) : (
        <ul className="divide-y divide-line">
          {examples.data.map((example) => (
            <li key={example.name}>
              <button
                type="button"
                disabled={load.isPending}
                onClick={() => pick('example', example.name)}
                className="w-full px-2 py-2 text-left hover:bg-surface-2"
              >
                <span className="block font-mono text-sm">{example.name}</span>
                <span className="block text-xs text-ink-3">{example.description}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Dialog>
  )
}
