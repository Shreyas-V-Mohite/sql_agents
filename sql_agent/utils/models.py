"""
Model utilities for loading Qwen models.
Supports both base model and fine-tuned LoRA adapters.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional


class QwenModel:
    """Wrapper for Qwen model with chat interface."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        lora_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto"
    ):
        """
        Initialize Qwen model.
        
        Args:
            model_name: Base model name/path
            lora_path: Path to LoRA adapter weights (optional)
            device_map: Device mapping strategy
            torch_dtype: Torch dtype for model
        """
        self.model_name = model_name
        self.lora_path = lora_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
        
        self.model.eval()
    
    def invoke(self, prompt: str, max_new_tokens: int = 1024) -> "QwenResponse":
        """
        Generate response for a prompt (LangChain-compatible interface).
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            QwenResponse object with .content attribute
        """
        messages = [
            {"role": "system", "content": "You are a helpful SQL assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for SQL generation
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return QwenResponse(content=response)
    
    def with_structured_output(self, schema):
        """
        Return a wrapper that parses output into a Pydantic model.
        
        Args:
            schema: Pydantic model class
            
        Returns:
            StructuredQwenModel wrapper
        """
        return StructuredQwenModel(self, schema)


class QwenResponse:
    """Response object mimicking LangChain message format."""
    
    def __init__(self, content: str):
        self.content = content


class StructuredQwenModel:
    """Wrapper for structured output parsing."""
    
    def __init__(self, model: QwenModel, schema):
        self.model = model
        self.schema = schema
    
    def invoke(self, prompt: str) -> dict:
        """
        Generate and parse structured output.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Parsed Pydantic model instance
        """
        import json
        import re
        
        # Add JSON instruction to prompt
        structured_prompt = f"""{prompt}

Respond with a valid JSON object matching this schema:
{self.schema.model_json_schema()}

Return ONLY the JSON object, no other text."""
        
        response = self.model.invoke(structured_prompt)
        content = response.content.strip()
        
        # Try to extract JSON from response
        # Handle cases where model wraps in markdown
        if "```json" in content:
            match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        elif "```" in content:
            match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        
        # Parse JSON
        try:
            data = json.loads(content)
            return self.schema(**data)
        except (json.JSONDecodeError, Exception) as e:
            # Try to find JSON object in response
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return self.schema(**data)
                except:
                    pass
            raise ValueError(f"Failed to parse structured output: {e}\nResponse: {content}")


# Global model cache to avoid reloading
_model_cache = {}


def get_qwen_model(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    lora_path: Optional[str] = None,
    use_cache: bool = True
) -> QwenModel:
    """
    Get or create a Qwen model instance.
    
    Args:
        model_name: Base model name
        lora_path: Optional LoRA adapter path
        use_cache: Whether to cache and reuse models
        
    Returns:
        QwenModel instance
    """
    cache_key = f"{model_name}:{lora_path or 'base'}"
    
    if use_cache and cache_key in _model_cache:
        return _model_cache[cache_key]
    
    model = QwenModel(model_name=model_name, lora_path=lora_path)
    
    if use_cache:
        _model_cache[cache_key] = model
    
    return model
