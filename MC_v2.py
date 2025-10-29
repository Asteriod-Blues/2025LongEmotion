import json
import ollama
import time
import os
from typing import List, Dict, Any

class OllamaCounselorProcessor:
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
    
    def process_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个测试样例
        
        Args:
            example: 单个测试样例的字典
            
        Returns:
            包含原始数据和模型回答的字典
        """
        # 构建完整的提示词 - 全部使用英文
        conversation_history = example.get("conversation_history", "")
        
        full_prompt = f"""
{self.base_prompt}

Conversation History:
{conversation_history}

Please act as a psychological counselor and provide an empathetic and context-aware emotional support response to the client based on the conversation history above.
Please ensure your response:
1. Shows understanding and empathy for the client's feelings
2. Is based on the specific context in the conversation history
3. Provides warm, supportive responses
4. Helps the client further explore their emotions and experiences
5. Maintains a professional and caring tone

Counselor's response:
"""
        
        try:
            # 调用Ollama模型
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt
            )
            
            # 提取模型回答
            model_response = response['response']
            
            # 构建结果 - 使用指定的输出格式
            result = {
                "id": example.get("id"),
                "predicted_response": model_response
            }
            
            return result
            
        except Exception as e:
            print(f"处理样例 {example.get('id')} 时出错: {str(e)}")
            # 出错时也保持相同的输出格式
            return {
                "id": example.get("id"),
                "predicted_response": f"Error: {str(e)}"
            }
    
    def process_jsonl_file(self, input_file: str, output_file: str = None, delay: float = 1.0):
        """
        处理整个JSONL文件
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出结果文件路径（可选）
            delay: 每次API调用之间的延迟（秒）
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
        
        total_examples = len(lines)
        print(f"找到 {total_examples} 个测试样例")
        
        # 处理每个样例
        for i, line in enumerate(lines):
            try:
                example = json.loads(line.strip())
                print(f"处理样例 {i+1}/{total_examples} (ID: {example.get('id')})...")
                
                result = self.process_single_example(example)
                self.results.append(result)
                
                # 显示进度
                print(f"✓ 完成样例 {example.get('id')}")
                
                # 添加延迟以避免过快的API调用
                if i < total_examples - 1:  # 不在最后一个样例后等待
                    time.sleep(delay)
                    
            except json.JSONDecodeError as e:
                print(f"❌ 解析第 {i+1} 行JSON时出错: {str(e)}")
                # 即使解析错误也创建一个结果条目
                self.results.append({
                    "id": i,
                    "predicted_response": f"JSON解析错误: {str(e)}"
                })
            except Exception as e:
                print(f"❌ 处理第 {i+1} 行时出错: {str(e)}")
                self.results.append({
                    "id": i,
                    "predicted_response": f"处理错误: {str(e)}"
                })
        
        # 保存结果
        if output_file:
            self.save_results(output_file)
        
        print(f"处理完成！共处理 {len(self.results)} 个样例")
    
    def save_results(self, output_file: str):
        """保存结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"结果已保存到: {output_file}")
    
    def print_summary(self):
        """打印处理摘要"""
        successful = len([r for r in self.results if not r.get("predicted_response", "").startswith("Error")])
        failed = len([r for r in self.results if r.get("predicted_response", "").startswith("Error")])
        
        print("\n" + "="*50)
        print("处理摘要")
        print("="*50)
        print(f"总样例数: {len(self.results)}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"使用模型: {self.model_name}")

# 使用示例
def main():
    # 配置参数
    MODEL_NAME = "MC"  # 替换为您想要使用的Ollama模型
    
    # 使用绝对路径
    BASE_DIR = "/home/zhuhaoyu/Psychology_test/MC"
    INPUT_FILE = os.path.join(BASE_DIR, "Conversations_Long.jsonl")  # 请根据实际文件名修改
    OUTPUT_FILE = os.path.join(BASE_DIR, "MCresults.jsonl")
    
    # 基础提示词 - 英文版本，针对心理咨询师角色定制
    BASE_PROMPT = """
You are a professional psychological counselor. Your task is to provide empathetic and context-aware emotional support to clients based on their conversation history.

As a counselor, you should:
1. Demonstrate deep understanding and empathy for the client's feelings and experiences
2. Provide warm, supportive, and non-judgmental responses
3. Help the client explore their emotions and experiences further
4. Acknowledge the client's progress and strengths
5. Maintain a professional yet caring tone
6. Build on the existing therapeutic relationship established in the conversation history
7. Offer appropriate guidance while respecting the client's autonomy

Remember to:
- Validate the client's emotions and experiences
- Use reflective listening techniques
- Ask open-ended questions to encourage deeper exploration
- Provide appropriate emotional support and encouragement
- Maintain appropriate professional boundaries

Your responses should be in English, as this is a professional counseling session.
"""
    
    # 创建处理器
    processor = OllamaCounselorProcessor(
        model_name=MODEL_NAME,
        base_prompt=BASE_PROMPT
    )
    
    # 处理文件
    processor.process_jsonl_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        delay=2.0  # 每次调用间隔2秒
    )
    
    # 显示摘要
    processor.print_summary()
    
    # 显示前几个结果作为示例
    if processor.results:
        print("\n前3个结果示例:")
        for i, result in enumerate(processor.results[:3]):
            print(f"\n--- 结果 {i+1} ---")
            print(f"ID: {result.get('id')}")
            print(f"预测回应: {result.get('predicted_response', 'N/A')[:200]}...")

if __name__ == "__main__":
    main()