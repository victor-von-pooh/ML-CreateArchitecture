import { useQuery } from '@tanstack/react-query'

import { useApi, type Issue } from '../../api/client'

/**
 * 登録されていない部品の誤りがある場合に, 自作の部品を起動の設定に足す案内を出す
 * (定義に書いた名前からは読み込まないので, 起動し直してもらう)
 */
export function CustomModuleHint({ issues }: { issues: Issue[] }) {
  const api = useApi()
  const unknown = issues.some((issue) => issue.code === 'unknown_unit')
  const modules = useQuery({
    queryKey: ['modules'],
    queryFn: api.listModules,
    staleTime: Infinity,
    enabled: unknown,
  })
  if (!unknown) return null
  const env = modules.data?.env ?? 'ML_CREATEARCHITECTURE_MODULES'
  const loaded = modules.data?.modules ?? []
  return (
    <div
      role="note"
      aria-label="自作の部品の足し方"
      className="space-y-1 rounded-md border border-line bg-surface-2 p-2 text-xs text-ink-2"
    >
      <p>
        自作の部品なら, 部品を宣言したモジュールの名前を起動の設定{' '}
        <code className="font-mono">{env}</code> (カンマ区切り) に足して,
        アプリを起動し直してください.
      </p>
      <p className="font-mono break-all text-ink-3">{env}=my_lab.units make serve</p>
      <p className="text-ink-3">
        今読み込んでいるモジュール:{' '}
        {modules.isPending
          ? '読んでいます'
          : modules.isError
            ? '読めませんでした'
            : loaded.length > 0
              ? loaded.map((module) => module.name).join(', ')
              : 'なし'}
      </p>
      <p className="text-ink-3">
        書いたモジュールは起動のときに任意のコードとして動くので,
        信頼できるものだけを書いてください. 定義に書かれた名前からは何も読み込みません.
      </p>
    </div>
  )
}
