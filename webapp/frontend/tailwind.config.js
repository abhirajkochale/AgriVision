/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        agri: {
          50: '#f2f8f4',
          100: '#e1f0e7',
          200: '#c5e2d2',
          300: '#9bceb3',
          400: '#6bb38f',
          500: '#469670',
          600: '#347a59',
          700: '#2a6148',
          800: '#234e3b',
          900: '#1d4132',
          950: '#0f241c',
        },
        earth: {
          50: '#fbf9f6',
          100: '#f4efe8',
          200: '#e8ded0',
          300: '#d7c5b0',
          400: '#c3a68d',
          500: '#b48e71',
          600: '#a5795c',
          700: '#89624b',
          800: '#715140',
          900: '#5c4337',
        }
      },
    },
  },
  plugins: [],
}
