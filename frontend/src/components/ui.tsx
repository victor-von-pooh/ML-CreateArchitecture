/**
 * モデルの作成・選定の段階に固有の部品. 共通の部品 (殻の正の写しの common.tsx)
 * もここから書き出す (部品の import の先を ui のままにできるため)
 */
import { useId } from 'react'

export {
  Button,
  Card,
  Dialog,
  EmptyState,
  ErrorBox,
  FieldError,
  PageHeader,
  Spinner,
  StatTile,
  Tabs,
} from './common'

/** 同梱の例を選ぶボタン (開くと部分モデルにするの一覧で使う). 雛形の説明は
 * 数行になる (学習の step で行うことまで書く) ので, 行の間を空けて全文を出し,
 * 長い語でも枠の外に出さない. 読み上げの名前は例の名前だけにし, 説明は
 * ボタンの説明として読む (一覧を移るたびに全文を名前として読まないため) */
export function ExampleButton({
  name,
  description,
  disabled,
  onClick,
}: {
  name: string
  description: string
  disabled?: boolean
  onClick: () => void
}) {
  const id = useId()
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      aria-labelledby={`${id}-name`}
      aria-describedby={description ? `${id}-description` : undefined}
      className="w-full px-2 py-2 text-left hover:bg-surface-2"
    >
      <span id={`${id}-name`} className="block font-mono text-sm">
        {name}
      </span>
      {description && (
        <span
          id={`${id}-description`}
          className="mt-0.5 block text-xs leading-relaxed wrap-anywhere text-ink-3"
        >
          {description}
        </span>
      )}
    </button>
  )
}
