import json
import ollama
import time
import re
from typing import List, Dict, Any

class OllamaJSONLProcessor:
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
    
    def extract_emotion_from_json(self, model_response: str) -> str:
        """
        从模型回答的JSON格式中提取情感
        
        Args:
            model_response: 模型的完整回答
            
        Returns:
            提取出的情感，如果无法提取则返回None
        """
        try:
            # 尝试从响应中提取JSON部分
            json_match = re.search(r'\{[^{}]*"Emotion"[^{}]*\}', model_response)
            if json_match:
                json_str = json_match.group()
                emotion_data = json.loads(json_str)
                return emotion_data.get("Emotion")
        except:
            pass
        return None
    
    def extract_emotion(self, model_response: str, choices: List[str]) -> str:
        """
        从模型回答中提取情感
        
        Args:
            model_response: 模型的完整回答
            choices: 可选的情感列表
            
        Returns:
            提取出的情感，强制从选项中选择一个
        """
        # 首先尝试从JSON格式中提取
        emotion_from_json = self.extract_emotion_from_json(model_response)
        if emotion_from_json and emotion_from_json in choices:
            return emotion_from_json
        
        # 将模型回答转换为小写以便匹配
        response_lower = model_response.lower()
        
        # 将选项也转换为小写
        choices_lower = [choice.lower() for choice in choices]
        
        # 尝试在回答中直接找到选项中的情感词
        for i, choice in enumerate(choices_lower):
            if choice in response_lower:
                return choices[i]  # 返回原始大小写的选项
        
        # 如果直接匹配失败，尝试查找常见的情感指示词
        emotion_patterns = {
            "Delight": [r"delight", r"happy", r"joy", r"pleasure"],
            "Anger": [r"anger", r"angry", r"mad", r"furious"],
            "Embarrassment": [r"embarrassment", r"embarrassed", r"ashamed", r"shame"],
            "Hopeless": [r"hopeless", r"despair", r"desperate", r"no hope"],
            "Pride": [r"pride", r"proud", r"accomplish", r"achievement"],
            "Disappointment": [r"disappointment", r"disappointed", r"let down", r"dissatisfied"]
        }
        
        for emotion, patterns in emotion_patterns.items():
            if emotion in choices:  # 只检查在选项中的情感
                for pattern in patterns:
                    if re.search(pattern, response_lower):
                        return emotion
        
        # 如果还是找不到，选择第一个选项作为默认
        return choices[0] if choices else "Delight"
    
    def process_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个测试样例
        
        Args:
            example: 单个测试样例的字典
            
        Returns:
            包含原始数据和模型回答的字典
        """
        # 构建完整的提示词
        context = example.get("context", "")
        subject = example.get("Subject", "")
        choices = example.get("choices", [])
        
        full_prompt = f"""
{self.base_prompt}

Scenario:
{context}

Question: What emotion(s) would {subject} ultimately feel in this situation?
Choices: {', '.join(choices)}

Only return the selected label in the output, without any additional content.
You MUST select one emotion from the provided choices.
Please provide your answer in a structured JSON format as follows: 
{{"Emotion": "selected_emotion"}}
"""
        try:
            # 调用Ollama模型
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt
            )
            
            # 提取模型回答
            model_response = response['response']
            
            # 从模型回答中提取情感
            predicted_emotion = self.extract_emotion(model_response, choices)
            
            # 构建结果 - 仅包含ID和预测情感
            result = {
                "id": example.get("id"),
                "predicted_emotion": predicted_emotion
            }
            
            return result
            
        except Exception as e:
            print(f"处理样例 {example.get('id')} 时出错: {str(e)}")
            # 出错时选择第一个选项作为默认
            choices = example.get("choices", [])
            default_emotion = choices[0] if choices else "Delight"
            return {
                "id": example.get("id"),
                "predicted_emotion": default_emotion
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
                print(f"处理样例 {i+1}/{total_examples} (ID: {example.get('id')})...")
                
                result = self.process_single_example(example)
                self.results.append(result)
                
                # 显示进度
                print(f"✓ 完成样例 {example.get('id')}: {result.get('predicted_emotion')}")
                
                # 添加延迟以避免过快的API调用
                if i < total_examples - 1:  # 不在最后一个样例后等待
                    time.sleep(delay)
                    
            except json.JSONDecodeError as e:
                print(f"❌ 解析第 {i+1} 行JSON时出错: {str(e)}")
                self.results.append({
                    "id": i,
                    "predicted_emotion": "Delight"  # 默认情感
                })
            except Exception as e:
                print(f"❌ 处理第 {i+1} 行时出错: {str(e)}")
                self.results.append({
                    "id": i,
                    "predicted_emotion": "Delight"  # 默认情感
                })
        
        # 保存结果
        if output_file:
            self.save_results(output_file)
        
        print(f"处理完成！共处理 {len(self.results)} 个样例")
    
    def save_results(self, output_file: str):
        """保存结果到文件，确保格式为指定的JSONL格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                # 只包含id和predicted_emotion两个字段
                output_item = {
                    "id": result.get("id"),
                    "predicted_emotion": result.get("predicted_emotion")
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        print(f"结果已保存到: {output_file}")
        
        # 显示输出文件的前几行作为验证
        print("\n输出文件前几行预览:")
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 3:  # 只显示前3行
                    print(f"  {line.strip()}")
                else:
                    break
    
    def print_summary(self):
        """打印处理摘要"""
        successful = len([r for r in self.results if r.get("predicted_emotion")])
        failed = len([r for r in self.results if not r.get("predicted_emotion")])
        
        print("\n" + "="*50)
        print("处理摘要")
        print("="*50)
        print(f"总样例数: {len(self.results)}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"使用模型: {self.model_name}")
        
        # 显示情感分布
        emotion_counts = {}
        for result in self.results:
            emotion = result.get("predicted_emotion")
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\n情感分布:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")

def main():
    # 配置参数
    MODEL_NAME = "testclassifi"  # 替换为您想要使用的Ollama模型
    INPUT_FILE = "Emotion_Classification.jsonl"  # 输入文件路径
    OUTPUT_FILE = "ECresults.jsonl"  # 输出文件路径
    
    # 基础提示词 - 英文版本，针对情感分析任务定制
    BASE_PROMPT = """
You are an expert in emotion analysis. Please carefully read the following scenario and analyze the emotional state of the specified character.

Task requirements:

1. Carefully analyze the descriptions of the subject character in the scenario
2. Consider the character's behavior, dialogue, inner thoughts, and other clues
3. You MUST select one emotion from the provided choices
4. Do not provide any explanation or reasoning in your response
5. Only output the selected emotion in the specified JSON format

Make sure your analysis is based on textual evidence, not subjective guesses.

Focus specifically on the emotional state of the subject character in the given scenario.
"""
    
    # 用户选择模式
    print("选择运行模式:")
    print("1. 完整数据集处理")
    print("2. 小单位数据集测试")
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "2":
        # 小单位数据集测试模式
        print("\n运行小单位数据集测试模式...")
        # 创建一个小测试数据集
        test_data = [
            {
                "id": 0,
                "context": "John just won the lottery and is celebrating with his friends.",
                "Subject": "John",
                "choices": ["Delight", "Anger", "Embarrassment"]
            },
            {
                "id": 1, 
                "context": "Mary failed her final exam despite studying hard for weeks.",
                "Subject": "Mary",
                "choices": ["Disappointment", "Pride", "Hopeless"]
            },
            {
                "id": 2,
                "context": "Tom was praised by his boss in front of the whole team for his excellent work.",
                "Subject": "Tom", 
                "choices": ["Pride", "Embarrassment", "Anger"]
            }
        ]
        
        # 保存测试数据到临时文件
        INPUT_FILE = "test_small_dataset.jsonl"
        with open(INPUT_FILE, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        max_examples = 3
        OUTPUT_FILE = "test_results.jsonl"
        print("创建小测试数据集完成！")
    else:
        # 完整数据集处理模式
        print("\n运行完整数据集处理模式...")
        max_examples = None  # 处理所有样例
    
    # 创建处理器
    processor = OllamaJSONLProcessor(
        model_name=MODEL_NAME,
        base_prompt=BASE_PROMPT
    )

    # 处理文件
    processor.process_jsonl_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        delay=2.0,  # 每次调用间隔2秒
        max_examples=max_examples
    )

    # 显示摘要
    processor.print_summary()

if __name__ == "__main__":
    main()