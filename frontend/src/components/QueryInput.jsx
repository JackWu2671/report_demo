import React, { useRef, useEffect } from 'react'

export default function QueryInput({ value, onChange, onSend, disabled }) {
  const ref = useRef(null)

  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = 'auto'
      ref.current.style.height = Math.min(ref.current.scrollHeight, 160) + 'px'
    }
  }, [value])

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      onSend()
    }
  }

  const canSend = !disabled && value.trim()

  return (
    <div className="query-input">
      <textarea
        ref={ref}
        className="query-input__textarea"
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="输入消息，Enter 发送，Shift+Enter 换行"
        disabled={disabled}
        rows={1}
      />
      <button
        className={`query-input__btn${canSend ? '' : ' query-input__btn--disabled'}`}
        onClick={onSend}
        disabled={!canSend}
      >
        {disabled ? (
          <span className="query-input__spinner" />
        ) : (
          <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
          </svg>
        )}
      </button>
    </div>
  )
}
