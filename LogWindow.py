class LogWindow():
    def log(self, time: str, step: int, winrate: float, epsilon: float) -> str:
        lines = [
            f"Logged at {time}",
            f"Step: {step}",
            f"Winrate: {winrate:.2f}",
            f"Epsilon: {epsilon:.4f}"
        ]
        return "\n".join(lines)
    
if __name__ == "__main__":
    lw = LogWindow()
    print(lw.log(1000000, 0.1, 0.5, 10, 5))
