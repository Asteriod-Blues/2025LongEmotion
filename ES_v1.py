import json
import requests
import time
from typing import List, Dict, Any
import logging
import re

class OllamaJSONLProcessor:
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
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, str]:
        """
        从模型回复中提取JSON格式内容
        """
        if not response_text:
            return {
                "predicted_cause": "ERROR: Empty response",
                "predicted_symptoms": "ERROR: Empty response",
                "predicted_treatment_process": "ERROR: Empty response",
                "predicted_illness_Characteristics": "ERROR: Empty response",
                "predicted_treatment_effect": "ERROR: Empty response"
            }
        
        # 尝试直接解析JSON
        try:
            # 查找JSON代码块
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                # 如果没有代码块标记，尝试直接解析整个响应
                return json.loads(response_text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提取各个字段
            self.logger.warning("JSON解析失败，尝试提取字段")
            return self._extract_fields_manually(response_text)
    
    def _extract_fields_manually(self, text: str) -> Dict[str, str]:
        """
        手动从文本中提取各个字段
        """
        result = {
            "predicted_cause": "",
            "predicted_symptoms": "",
            "predicted_treatment_process": "",
            "predicted_illness_Characteristics": "",
            "predicted_treatment_effect": ""
        }
        
        # 简单的关键词匹配提取
        patterns = {
            "predicted_cause": [r'"predicted_cause"\s*:\s*"([^"]*)"', r'causes?["\']?\s*:\s*["\']?([^"\']*)'],
            "predicted_symptoms": [r'"predicted_symptoms"\s*:\s*"([^"]*)"', r'symptoms?["\']?\s*:\s*["\']?([^"\']*)'],
            "predicted_treatment_process": [r'"predicted_treatment_process"\s*:\s*"([^"]*)"', r'treatment.process["\']?\s*:\s*["\']?([^"\']*)'],
            "predicted_illness_Characteristics": [r'"predicted_illness_Characteristics"\s*:\s*"([^"]*)"', r'characteristics["\']?\s*:\s*["\']?([^"\']*)'],
            "predicted_treatment_effect": [r'"predicted_treatment_effect"\s*:\s*"([^"]*)"', r'treatment.effect["\']?\s*:\s*["\']?([^"\']*)']
        }
        
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()
                    break
            if not result[field]:
                result[field] = "未提取到相关信息"
        
        return result
    
    def build_prompt(self, case_data: Dict[str, Any]) -> str:
        """
        构建完整的提示词
        """
        # 提取数据字段
        case_description = case_data.get("case_description", [])
        consultation_process = case_data.get("consultation_process", [])
        experience_and_reflection = case_data.get("experience_and_reflection", "")
        
        # 将列表转换为字符串
        case_desc_str = " ".join(case_description) if isinstance(case_description, list) else str(case_description)
        consultation_str = " ".join(consultation_process) if isinstance(consultation_process, list) else str(consultation_process)
        
        # 构建完整提示词
        prompt = f"""你是一名心理咨询专家。请基于以下心理咨询报告内容进行分析：

案例描述: {case_desc_str}
咨询过程: {consultation_str}
经验与反思: {experience_and_reflection}

请根据提供的内容总结以下信息：
- 原因：个体心理问题的潜在或直接原因
- 症状：个体表现出的自我报告或可观察的生理、心理或行为症状
- 治疗过程：在咨询过程中应用的心理治疗方法、技术和阶段性干预
- 疾病特征：心理问题的关键特征或发展模式
- 治疗效果：治疗的影响或结果，包括个体状况的变化

请严格按照以下JSON格式提供你的分析，不要添加任何其他文本：
{{
    "predicted_cause": "...",
    "predicted_symptoms": "...", 
    "predicted_treatment_process": "...",
    "predicted_illness_Characteristics": "...",
    "predicted_treatment_effect": "..."
}}"""
        
        return prompt
    
    def process_single_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个测试用例
        """
        case_id = case_data.get("id", "unknown")
        self.logger.info(f"处理用例 ID: {case_id}")
        
        # 构建提示词
        prompt = self.build_prompt(case_data)
        
        # 获取模型回复
        response = self.generate_response(prompt)
        
        # 提取JSON内容
        extracted_data = self.extract_json_from_response(response)
        
        # 构建最终输出格式
        result = {
            "id": case_id,
            "predicted_cause": extracted_data.get("predicted_cause", ""),
            "predicted_symptoms": extracted_data.get("predicted_symptoms", ""),
            "predicted_treatment_process": extracted_data.get("predicted_treatment_process", ""),
            "predicted_illness_Characteristics": extracted_data.get("predicted_illness_Characteristics", ""),
            "predicted_treatment_effect": extracted_data.get("predicted_treatment_effect", "")
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
                    
                    # 处理单个用例
                    result = self.process_single_case(case_data)
                    
                    # 写入结果（严格符合要求格式）
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                    outfile.flush()
                    
                    processed_count += 1
                    self.logger.info(f"成功处理第 {line_num} 行，ID: {case_data.get('id', 'unknown')}")
                    
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
    parser = argparse.ArgumentParser(description='使用Ollama处理JSONL心理学测试文件')
    parser.add_argument('--input', '-i', required=True, help='输入JSONL文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSONL文件路径')
    parser.add_argument('--model', '-m', default='llama2', help='Ollama模型名称 (默认: llama2)')
    parser.add_argument('--url', '-u', default='http://localhost:11434', help='Ollama服务器URL')
    parser.add_argument('--test', '-t', action='store_true', help='测试模式：只处理前3条数据')
    
    args = parser.parse_args()
    
    # 初始化处理器
    processor = OllamaJSONLProcessor(
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
            print("\n前几条输出结果:")
            count = 0
            for line in f:
                if count >= 3:  # 只显示前3条
                    break
                print(line.strip())
                count += 1
    except FileNotFoundError:
        print("输出文件未找到")

if __name__ == "__main__":
    main()