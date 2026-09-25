/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        paper: '#F6F1E8', surface: '#FFFCF6', ink: '#211C17', muted: '#756A5F', line: '#DDD1C3',
        orange: '#E86F21', rust: '#B94D0D', ochre: '#F2B84B', danger: '#C74C36', good: '#4F7A59', sand: '#EEE5D9',
      },
      fontFamily: { sans: ['DM Sans', 'Arial', 'sans-serif'], serif: ['Fraunces', 'Georgia', 'serif'], mono: ['DM Mono', 'monospace'] },
    },
  },
  plugins: [],
}
