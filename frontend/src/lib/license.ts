// 読む重みを決める引数 (これと重みを読む引数だけが変わったときに聞き直す). 名前と版と,
// 名前と版以外で重みを変える引数 (ML-ModelControl の pretrained.py の WEIGHT_OVERRIDE_PARAMS
// と timm の pretrained_cfg. 足したら合わせる)
export const SOURCE_PARAMS = [
  'model_name',
  'revision',
  'name',
  'encoder_name',
  'pretrained_cfg',
  'pretrained_cfg_overlay',
  'checkpoint_path',
  'pretrained_url',
  'weights_url',
  'pretrained_path',
]

/** ライセンスに関わる引数だけを, 決まった順の JSON の文字にする (落ち着くまで待つ値を安定させる) */
export function licenseParams(params: Record<string, unknown>, weightsParams: string[]): string {
  const keys = [...new Set([...SOURCE_PARAMS, ...weightsParams])].sort()
  return JSON.stringify(keys.filter((key) => key in params).map((key) => [key, params[key]]))
}

/** 重みの大きさのバイト数を, 人が読める単位 (B, KB, MB, GB, TB. 1024 ごと) の文字にする.
 * 負の値, 無限, NaN は null */
export function formatBytes(size: number): string | null {
  if (!Number.isFinite(size) || size < 0) return null
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  // バイトと 10 以上は整数, 他は小数 1 桁に丸める
  const round = (value: number, index: number) =>
    index === 0 || value >= 10 ? Math.round(value) : Math.round(value * 10) / 10
  // 単位を上げるかは丸めた後の値で決める (1048575 B を 1024 KB にしないため)
  let index = 0
  let value = size
  let rounded = round(value, index)
  while (rounded >= 1024 && index < units.length - 1) {
    value /= 1024
    index += 1
    rounded = round(value, index)
  }
  // 丸めて 10 になったもの (9.97 MB) は整数で出す
  const text = index === 0 || rounded >= 10 ? rounded.toString() : rounded.toFixed(1)
  return `${text} ${units[index]}`
}
