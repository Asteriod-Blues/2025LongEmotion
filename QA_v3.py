import json
import ollama
import time
import os
import re
from typing import List, Dict, Any

class OllamaLiteratureProcessor:
    def __init__(self, model_name: str, base_prompt: str):
        """
        初始化处理器
        
        Args:
            model_name: Ollama中的模型名称
            base_prompt: 整体的基础提示词
        """
        self.model_name = model_name
        self.base_prompt = base_prompt
        self.results = []
    
    def is_yes_no_question(self, question: str) -> bool:
        """
        判断问题是否为yes/no类型的问题
        
        Args:
            question: 问题文本
            
        Returns:
            如果是yes/no问题返回True，否则返回False
        """
        # 匹配常见的是非疑问词开头的问题
        yes_no_patterns = [
            r'^is\s+', r'^are\s+', r'^does\s+', r'^do\s+', r'^did\s+', 
            r'^was\s+', r'^were\s+', r'^has\s+', r'^have\s+', r'^had\s+',
            r'^can\s+', r'^could\s+', r'^will\s+', r'^would\s+', r'^should\s+',
            r'^must\s+', r'^may\s+', r'^might\s+'
        ]
        
        question_lower = question.strip().lower()
        
        # 检查是否以疑问词开头
        for pattern in yes_no_patterns:
            if re.match(pattern, question_lower):
                return True
        
        # 检查是否包含明确的yes/no指示
        if any(phrase in question_lower for phrase in [
            ' yes or no', ' true or false', ' correct or incorrect'
        ]):
            return True
            
        return False
    
    def clean_response(self, response: str, is_yes_no: bool) -> str:
        """
        清理模型响应，确保符合输出要求
        
        Args:
            response: 原始模型响应
            is_yes_no: 是否为yes/no问题
            
        Returns:
            清理后的响应
        """
        response = response.strip()
        
        if is_yes_no:
            # 对于yes/no问题，只保留yes或no
            response_lower = response.lower()
            
            # 匹配yes/no及其变体
            if re.match(r'^\s*yes\s*$', response_lower, re.IGNORECASE):
                return "yes"
            elif re.match(r'^\s*no\s*$', response_lower, re.IGNORECASE):
                return "no"
            else:
                # 如果响应不是纯yes/no，尝试提取第一个单词
                first_word = response.split()[0].lower() if response.split() else ""
                if first_word in ['yes', 'no']:
                    return first_word
                else:
                    # 如果无法提取，返回原始响应的前几个字符（限制长度）
                    return response[:50]  # 限制长度避免过长输出
        
        # 对于非yes/no问题，返回原样但限制长度
        return response[:200]  # 限制长度避免过长输出
    
    def process_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个测试样例
        
        Args:
            example: 单个测试样例的字典
            
        Returns:
            包含ID和预测答案的字典
        """
        # 构建完整的提示词 - 使用更严格的提示词
        context = example.get("context", "")
        problem = example.get("problem", "")
        
        # 判断是否为yes/no问题
        is_yes_no = self.is_yes_no_question(problem)
        
        # 根据问题类型构建不同的提示词
        if is_yes_no:
            answer_instruction = "Answer with exactly one word: 'yes' or 'no'. Do not include any other text or explanation."
        else:
            answer_instruction = "Answer concisely in one sentence. Do not provide any explanation or additional context."
        
        full_prompt = f"""
{self.base_prompt}

Article: {context}

Question: {problem}

{answer_instruction}
Answer:
"""
        
        try:
            # 调用Ollama模型
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt
            )
            
            # 提取模型回答并清理
            model_response = response['response'].strip()
            cleaned_response = self.clean_response(model_response, is_yes_no)
            
            # 构建结果 - 仅包含ID和预测答案
            result = {
                "id": example.get("id", 0),
                "predicted_answer": cleaned_response,
                "is_yes_no_question": is_yes_no,
                "original_response": model_response  # 保留原始响应用于调试
            }
            
            return result
            
        except Exception as e:
            print(f"处理样例 {example.get('id')} 时出错: {str(e)}")
            return {
                "id": example.get("id", 0),
                "predicted_answer": f"Error: {str(e)}",
                "is_yes_no_question": False,
                "original_response": ""
            }
    
    def process_jsonl_file(self, input_file: str, output_file: str = None, delay: float = 1.0, max_examples: int = None):
        """
        处理整个JSONL文件
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出结果文件路径（可选）
            delay: 每次API调用之间的延迟（秒）
            max_examples: 最大处理样例数（可选）
        """
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：找不到输入文件 {input_file}")
            print("请检查文件路径是否正确")
            return
        
        print(f"开始处理文件: {input_file}")
        print(f"使用模型: {self.model_name}")
        
        # 读取JSONL文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 限制处理数量
        if max_examples:
            lines = lines[:max_examples]
            print(f"测试模式: 只处理前 {max_examples} 个样例")
        
        total_examples = len(lines)
        print(f"找到 {total_examples} 个测试样例")
        
        # 处理每个样例
        for i, line in enumerate(lines):
            try:
                example = json.loads(line.strip())
                # 如果没有ID字段，使用索引作为ID
                if "id" not in example:
                    example["id"] = i
                
                print(f"处理样例 {i+1}/{total_examples} (ID: {example.get('id')})...")
                
                result = self.process_single_example(example)
                self.results.append(result)
                
                # 显示进度和结果
                is_yes_no = result.get("is_yes_no_question", False)
                answer_type = "yes/no" if is_yes_no else "open"
                print(f"✓ 完成样例 {example.get('id')} ({answer_type}): {result.get('predicted_answer')}")
                
                # 添加延迟以避免过快的API调用
                if i < total_examples - 1:  # 不在最后一个样例后等待
                    time.sleep(delay)
                    
            except json.JSONDecodeError as e:
                print(f"❌ 解析第 {i+1} 行JSON时出错: {str(e)}")
                self.results.append({
                    "id": i,
                    "predicted_answer": f"JSON解析错误: {str(e)}",
                    "is_yes_no_question": False,
                    "original_response": ""
                })
            except Exception as e:
                print(f"❌ 处理第 {i+1} 行时出错: {str(e)}")
                self.results.append({
                    "id": i,
                    "predicted_answer": f"处理错误: {str(e)}",
                    "is_yes_no_question": False,
                    "original_response": ""
                })
        
        # 保存结果
        if output_file:
            self.save_results(output_file)
        
        print(f"处理完成！共处理 {len(self.results)} 个样例")
    
    def save_results(self, output_file: str):
        """保存结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                # 只保存必要字段到输出文件
                output_result = {
                    "id": result.get("id"),
                    "predicted_answer": result.get("predicted_answer")
                }
                f.write(json.dumps(output_result, ensure_ascii=False) + '\n')
        print(f"结果已保存到: {output_file}")
    
    def print_summary(self):
        """打印处理摘要"""
        successful = len([r for r in self.results if not r.get("predicted_answer", "").startswith("Error")])
        failed = len([r for r in self.results if r.get("predicted_answer", "").startswith("Error")])
        yes_no_questions = len([r for r in self.results if r.get("is_yes_no_question", False)])
        
        print("\n" + "="*50)
        print("处理摘要")
        print("="*50)
        print(f"总样例数: {len(self.results)}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"yes/no问题: {yes_no_questions}")
        print(f"开放性问题: {len(self.results) - yes_no_questions}")
        print(f"使用模型: {self.model_name}")

# 使用示例
def main():
    # 配置参数
    MODEL_NAME = "QA"  # 替换为您想要使用的Ollama模型
    
    # 使用绝对路径
    BASE_DIR = "/home/zhuhaoyu/Psychology_test/QA/"
    INPUT_FILE = os.path.join(BASE_DIR, "Emotion%20QA.jsonl")  # 请根据实际文件名修改
    OUTPUT_FILE = os.path.join(BASE_DIR, "QAresults.jsonl")
    
    # 基础提示词 - 更严格的版本
    BASE_PROMPT = """
You are an AI assistant specialized in answering questions based on scientific articles in psychology and related fields.

Task: Answer questions based ONLY on the information in the provided article.

CRITICAL INSTRUCTIONS:
1. Read the provided scientific article carefully
2. Answer based ONLY on the information in the article
3. For yes/no questions: Answer with exactly one word - "yes" or "no". Do NOT include any other text, explanation, or punctuation.
4. For other questions: Answer concisely in one sentence. Do NOT provide any explanation or additional context.
5. Do NOT use information from outside the provided article
6. Do NOT add any reasoning, examples, or supporting statements

Remember: Your response must be solely based on the information contained in the provided article and must follow the format instructions exactly.
"""
    
    # 创建处理器
    processor = OllamaLiteratureProcessor(
        model_name=MODEL_NAME,
        base_prompt=BASE_PROMPT
    )
    
    # 处理文件 - 可以设置max_examples来限制处理数量
    processor.process_jsonl_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        delay=2.0,  # 每次调用间隔2秒
        max_examples=None  # 设置为None处理所有样例，设置为数字则处理前N个
    )
    
    # 显示摘要
    processor.print_summary()
    
    # 显示所有结果
    print("\n所有结果:")
    for i, result in enumerate(processor.results):
        answer_type = "yes/no" if result.get("is_yes_no_question") else "open"
        print(f"ID: {result.get('id')} ({answer_type}) -> 预测答案: {result.get('predicted_answer', 'N/A')}")

if __name__ == "__main__":
    main()