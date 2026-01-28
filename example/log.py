class LoggerWriter:
    """将标准输出重定向到日志"""
    def __init__(self, level):
        self.level = level
        self.buffer = ''
        
    def write(self, message):
        if message.strip():
            # 检查是否包含特殊字符（如进度条）
            if '\r' in message or '\b' in message:
                # 处理进度条等特殊输出
                self.buffer = message.strip('\r\n\b')
                if self.buffer:
                    self.level(self.buffer)
            else:
                self.level(message)
                
    def flush(self):
        pass