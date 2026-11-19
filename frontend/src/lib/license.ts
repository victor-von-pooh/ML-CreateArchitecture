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
