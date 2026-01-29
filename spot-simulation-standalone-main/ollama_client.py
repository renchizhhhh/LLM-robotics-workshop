import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://192.168.0.128:11434", timeout=5):
        """
        Initialize the Ollama client and verify the server is reachable.

        Args:
            base_url (str): The base URL of the Ollama server (default: http://192.168.0.128:11434)
            timeout (float): Timeout in seconds to use when checking connectivity (default: 2)

        Raises:
            ConnectionError: If the base_url is not reachable within the timeout.
        """
        # normalize base_url to avoid accidental double slashes later
        self.base_url = base_url.rstrip('/')
        self.model = "gpt-oss:120b"
        self._timeout = timeout

        try:
            resp = requests.get(self.base_url, timeout=self._timeout)
            if resp.status_code >= 500:
                raise ConnectionError(
                    f"Ollama server at {self.base_url} responded with status {resp.status_code}"
                )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server at {self.base_url}: {e}")

    def generate(self, prompt, stream=False, options=None):
        """
        Generate a response from the Ollama model.

        Args:
            prompt (str): The input prompt to send to the model
            stream (bool): Whether to stream the response (default: False)
            options (dict): Additional options for the model (optional)

        Returns:
            str or dict: The generated response
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": '40m'
        }

        if options:
            payload.update(options)

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()

            # Return the raw JSON so callers can handle different API shapes
            # (some Ollama versions return different keys).
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama server: {e}")
            return None

    def chat(self, messages, stream=False, options=None):
        """
        Send a chat message to the Ollama model.

        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            stream (bool): Whether to stream the response (default: False)
            options (dict): Additional options for the model (optional)

        Returns:
            str or dict: The generated response
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }

        if options:
            payload.update(options)

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()

            # Return the raw JSON for callers to extract message content as needed
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama server: {e}")
            return None

    def list_models(self):
        """
        List available models on the Ollama server.

        Returns:
            list: List of available models
        """
        url = f"{self.base_url}/api/tags"

        try:
            response = requests.get(url)
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize the client
    client = OllamaClient()

    # Example 1: Simple text generation
    print("Example 1: Text Generation")
    prompt = "Explain the concept of machine learning in simple terms."
    response = client.generate(prompt)
    if response:
        print(f"Response: {response}")
    print()

    # Example 2: Chat-style interaction
    print("Example 2: Chat Interaction")
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = client.chat(messages)
    if response:
        print(f"Response: {response}")
    print()

    # Example 3: List available models
    print("Example 3: Available Models")
    models = client.list_models()
    if models:
        print("Available models:")
        for model in models:
            print(f"- {model['name']}")
    else:
        print("No models found or unable to connect to server.")