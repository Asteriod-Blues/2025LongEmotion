import json
import requests
import time
from typing import List, Dict, Any
import logging
import re

class OllamaEmotionProcessor:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        初始化Ollama处理器
        """
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def check_ollama_connection(self) -> bool:
        """检查Ollama连接是否正常"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"连接Ollama失败: {e}")
            return False
    
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """
        使用Ollama生成回复
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.generate_url, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "").strip()
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"第 {attempt + 1} 次请求失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    self.logger.error(f"所有重试均失败: {e}")
                    return ""
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON解析错误: {e}")
                return ""
        
        return ""
    
    def extract_index_from_response(self, response_text: str) -> int:
        """
        从模型回复中提取索引值
        """
        if not response_text:
            self.logger.warning("模型返回空响应")
            return -1
        
        # 尝试提取JSON格式的index
        try:
            # 查找JSON代码块
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                return data.get("index", -1)
            else:
                # 如果没有代码块标记，尝试直接解析整个响应
                data = json.loads(response_text)
                return data.get("index", -1)
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试直接提取数字
            self.logger.warning("JSON解析失败，尝试直接提取数字")
            return self._extract_index_manually(response_text)
    
    def _extract_index_manually(self, text: str) -> int:
        """
        手动从文本中提取索引值
        """
        # 尝试匹配数字
        patterns = [
            r'"index"\s*:\s*(\d+)',
            r'index\s*:\s*(\d+)',
            r'不一致.*?(\d+)',
            r'inconsistent.*?(\d+)',
            r'index\s*=\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        # 如果找不到明确的数字，尝试提取第一个数字
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        
        self.logger.warning(f"无法从响应中提取索引: {text}")
        return -1
    
    def build_prompt(self, case_data: Dict[str, Any]) -> str:
        """
        构建完整的提示词
        """
        # 提取文本列表
        texts = case_data.get("text", [])
        num_texts = len(texts)
        
        # 构建文本列表字符串
        texts_str = ""
        for item in texts:
            index = item.get("index", -1)
            context = item.get("context", "")
            texts_str += f"Index {index}: {context}\n\n"
        
        # 构建完整提示词
        prompt = f"""There are {num_texts} texts in the text list.

Text list:
{texts_str}

Please identify the text that has an inconsistent emotion compared to the others and provide its index.

Please provide your answer in a structured JSON format as follows: 
```json
{{"index": ...}}
```"""
        
        return prompt
    
    def process_single_case(self, case_data: Dict[str, Any], case_id: int) -> Dict[str, Any]:
        """
        处理单个测试用例
        
        Args:
            case_data: 测试用例数据
            case_id: 测试用例ID
        """
        self.logger.info(f"处理用例 ID: {case_id}")
        
        # 构建提示词
        prompt = self.build_prompt(case_data)
        
        # 获取模型回复
        response = self.generate_response(prompt)
        
        # 提取索引值
        predicted_index = self.extract_index_from_response(response)
        
        # 构建最终输出格式
        result = {
            "id": case_id,
            "predicted_index": predicted_index
        }
        
        return result
    
    def process_jsonl_file(self, input_file: str, output_file: str, max_lines: int = None):
        """
        处理整个JSONL文件
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出JSONL文件路径
            max_lines: 最大处理行数，None表示处理所有行
        """
        # 检查连接
        if not self.check_ollama_connection():
            self.logger.error("无法连接到Ollama服务，请检查服务是否启动")
            return
        
        self.logger.info(f"开始处理文件: {input_file}")
        self.logger.info(f"使用模型: {self.model}")
        if max_lines:
            self.logger.info(f"最大处理行数: {max_lines}")
        
        processed_count = 0
        error_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                if max_lines and line_num > max_lines:
                    self.logger.info(f"达到最大行数限制 {max_lines}，停止处理")
                    break
                    
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON
                    case_data = json.loads(line)
                    
                    # 使用行号作为ID（从0开始）
                    case_id = line_num - 1
                    
                    # 处理单个用例
                    result = self.process_single_case(case_data, case_id)
                    
                    # 写入结果（严格符合要求格式）
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                    
                    processed_count += 1
                    self.logger.info(f"成功处理第 {line_num} 行，ID: {case_id}, 预测索引: {result['predicted_index']}")
                    
                    # 添加延迟避免过载
                    time.sleep(1)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"第 {line_num} 行JSON解析错误: {e}")
                    error_count += 1
                except Exception as e:
                    self.logger.error(f"处理第 {line_num} 行时发生错误: {e}")
                    error_count += 1
        
        self.logger.info(f"处理完成! 成功: {processed_count}, 失败: {error_count}")

def main():
    """主函数"""
    import argparse
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用Ollama处理情感分析JSONL文件')
    parser.add_argument('--input', '-i', required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSONL文件路径')
    parser.add_argument('--model', '-m', default='llama2', help='Ollama模型名称 (默认: llama2)')
    parser.add_argument('--url', '-u', default='http://localhost:11434', help='Ollama服务器URL')
    parser.add_argument('--test', '-t', action='store_true', help='测试模式：只处理前3条数据')
    
    args = parser.parse_args()
    
    # 初始化处理器
    processor = OllamaEmotionProcessor(
        base_url=args.url,
        model=args.model
    )
    
    # 处理文件
    if args.test:
        print("测试模式：只处理前3条数据")
        processor.process_jsonl_file(args.input, args.output, max_lines=3)
    else:
        print("完整模式：处理所有数据")
        processor.process_jsonl_file(args.input, args.output)
    
    print(f"\n处理完成！结果保存在: {args.output}")
    
    # 显示前几条结果
    try:
        with open(args.output, 'r', encoding='utf-8') as f:
            print("\n输出结果:")
            for line in f:
                print(line.strip())
    except FileNotFoundError:
        print("输出文件未找到")

if __name__ == "__main__":
    main()