import { useState, useRef, useEffect } from "react";
import { CornerDownLeft } from "lucide-react";

interface TerminalPromptProps {
  onCommand: (command: string) => void;
  disabled?: boolean;
}

export function TerminalPrompt({ onCommand, disabled }: TerminalPromptProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onCommand(input);
    setInput("");
  };

  // Auto-focus logic
  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 p-2 bg-card/30 border-t border-border backdrop-blur-sm">
      <span className="text-primary font-bold animate-pulse">{">"}</span>
      <input
        ref={inputRef}
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        disabled={disabled}
        className="flex-1 bg-transparent border-none outline-none text-foreground font-mono placeholder:text-muted-foreground/50"
        placeholder={disabled ? "PROCESSING..." : "Enter command..."}
        autoComplete="off"
        spellCheck="false"
      />
      <button 
        type="submit" 
        disabled={disabled || !input.trim()}
        className="text-primary/70 hover:text-primary transition-colors disabled:opacity-30"
      >
        <CornerDownLeft size={16} />
      </button>
    </form>
  );
}
