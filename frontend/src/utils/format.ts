export const integer = (value: number | null | undefined) => value == null ? 'No data' : new Intl.NumberFormat('en-US').format(value)
export const decimal = (value: number | null | undefined, digits = 2) => value == null ? 'No data' : value.toFixed(digits)
export const percent = (value: number | null | undefined, digits = 1) => value == null ? 'No data' : `${(value * 100).toFixed(digits)}%`
export const label = (value: string | null | undefined) => value ? value === 'ui_ux' ? 'UI / UX' : value.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown'
export const languageName = (code: string | null | undefined) => {
  if (!code || code === 'und') return 'Undetermined'
  try { return new Intl.DisplayNames(['en'], { type: 'language' }).of(code) ?? code.toUpperCase() }
  catch { return code.toUpperCase() }
}
