#!/usr/bin/env python

import re
from pathlib import Path
import subprocess
import socket
import click
from typing import Any, List, Tuple, Type, Optional
import langroid as lr
from langroid.agent.tools.orchestration import ForwardTool #, AgentDoneTool
from langroid.language_models.base import LLMResponse
from langroid.mytypes import Entity
Responder = Entity | Type["Task"]

# helper function to strip unspeakable characters from generated text
IGNORED_RESPONSE_SUFFIXES = [
    r'\s*\[Silence\]$',
    r'\s*<\[Silence\]>$',
    r'\s*<\|system.*$',
    r'\s*<\|user\|>.*$',
    r'\s*\[TOOL.*$',
    r'\s*<$',
    r'\s*$',
]

def strip_suffixes(text, suffixes):
    for suffix in suffixes:
        if re.search(suffix, text):
            text = re.sub(suffix, '', text)
            #print(f"suffix {suffix} -> [{text}]")
    return text

# TTS client call 
def speak_llm_response(response: str):
    verbal_response = strip_suffixes(response, IGNORED_RESPONSE_SUFFIXES)
    if verbal_response:
        subprocess.call([f'{Path(__file__).resolve(strict=True).parent}/../streaming_xtts_unity/streaming_tts_client.py', 
            '-p', 
            f'{verbal_response}'])

# ASR listener
host = 'localhost'
port = 27400
asr_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    asr_socket.connect((host, port))
except:
    print(f"Could not connect to ASR server on {host}:{port}")

def listen_for_user_speech():
    user_text = ""
    while True:
        data = asr_socket.recv(1024)
        decoded_data = ""
        if data:
            decoded_data = data.decode('utf-8')
            print(f"{decoded_data}")
            break
    return decoded_data[1:-1]

# External IVI listener call
ivi_ip = '10.10.10.158'
ivi_port = 4200

def send_ivi_command(event_name, **kwargs):
    command = [
        f'{Path(__file__).resolve(strict=True).parent}/../VAMPCar/scripts/ivi_interaction/sender2.py',
        f'--port={ivi_port}',
        f'--ip={ivi_ip}',
        f'--EventName={event_name}'
    ]
    for key, value in kwargs.items():
        command.append(f'--{key}={value}')
    print(f'~~> {" ".join(command)}')
    result = subprocess.run(command)
    if result.returncode:
        print("This tool returned an error.")
    #result = subprocess.call(command)
    #print(f"command returned {result}")


####### TOOLS #######

# Tool specifications

class HVACTool(lr.ToolMessage):
    request: str = "set_fan_speed"
    purpose: str = "To cool the cabin"
    speed: str

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        # Examples that will be compiled into few-shot examples for the LLM.
        # Each example can either be...
        return [
            # ... just instances of the tool-class, OR
            cls(speed="Low"),
            cls(speed="High"),
            (  # ...a tuple of "thought leading to tool", and the tool instance
                "I want to cool the cabin",
                cls(speed="High"),
            ),
            (
                "I want to stop cooling the cabin",
                cls(speed="Low"),
            ),
        ]
    
    def handle(self): # -> AgentDoneTool:
        print(f"~> Calling set_fan_speed({self.speed})")
        send_ivi_command("HVACCommand", HVACValue=self.speed)
        return f"Set fan speed to {self.speed}."
        
class AudioTool(lr.ToolMessage):
    request: str = "adjust_audio_volume"
    purpose: str = "To change the volume of the music within the range of 0 (silent) to 40 (maximum volume)"
    volume: int

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        # Examples that will be compiled into few-shot examples for the LLM.
        # Each example can either be...
        return [
            # ... just instances of the tool-class, OR
            cls(volume=10),
            (  # ...a tuple of "thought leading to tool", and the tool instance
                "I want to crank up the volume",
                cls(volume=35),
            ),
            (
                "I want to restore the volume to a normal level",
                cls(volume=10),
            ),
            (
                "I want to lower the volume",
                cls(volume=5)
            )
        ]

    def handle(self): # -> AgentDoneTool:
        print(f"~> Calling adjust_audio_volume({self.volume})")
        send_ivi_command("AudioCommand", Volume=self.volume)
        return f"Set volume level to {self.volume}."

class RadioTool(lr.ToolMessage):
    request: str = "search_for_radio_station"
    purpose: str = "To find a radio station given a search term"
    search_term: str

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        # Examples that will be compiled into few-shot examples for the LLM.
        # Each example can either be...
        return [
            (  
                "I want to find and play a station that plays classic rock music",
                cls(search_term="classic rock"),
            ),
            (
                "I want to find and play a station that features the hip hop genre",
                cls(search_term="hip hop"),
            ),
            (
                "I want to play National Public Radio",
                cls(search_term="NPR"),
            ),
            (
                "I want to play Kiss FM",
                cls(search_term="Kiss"),
            ),
        ]
    
    def handle(self): # -> AgentDoneTool:
        print(f"~> Calling search_for_radio_station({self.search_term})")
        send_ivi_command("RadioCommand", Query=self.search_term)
        return f"Searched for radio station related to {self.search_term} and found it."

####### AGENT + CONVERSATIONAL TASK #######

class CarCompanionAgent(lr.ChatAgent):

    USER_GOODBYE_PHRASE = "goodbye"
    
    def __init__(self, *args, **kwargs) -> None:
        # pull out added keywords (not part of ChatAgent class)
        self.voice_input = kwargs['voice_input']
        del kwargs['voice_input']
        self.speech_output = kwargs['speech_output']
        del kwargs['speech_output']
        # call ChatAgent constructor
        super().__init__(*args, **kwargs)
        self.user_said_goodbye: bool = False
    
    def handle_message_fallback(self, msg: str | lr.ChatDocument) -> Any:
        """
        We'd be here if there were no recognized tools in the incoming msg.
        If this was from LLM, forward to user.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent="User")

    def user_response(
        self,
        msg: Optional[str | lr.ChatDocument] = None,
    ) -> Optional[lr.ChatDocument]:
        """
        Overloading the default langroid function that gets the user's 
        response to current message. 
        Args:
            msg (str|ChatDocument): the string to respond to.
        Returns:
            (str) User response, packaged as a ChatDocument
        """
        if not self.voice_input:    
            return super().user_response(msg)

        else:
            if not self.user_can_respond(msg):
                return None

            if self.default_human_response is not None:
                user_msg = self.default_human_response
            else:
                print("\nCustomer: ", end="")
                user_msg = listen_for_user_speech()
                if user_msg.lower().replace(" ","") == self.USER_GOODBYE_PHRASE:
                    self.user_said_goodbye = True

            return self._user_response_final(msg, user_msg)


    def _render_llm_response(
        self, response: lr.ChatDocument | LLMResponse, citation_only: bool = False
    ) -> None:
        """
        Overloading the default langroid function that renders the LLM's response,
        adding a TTS output for any non-tool-call response portion 
        """
        super()._render_llm_response(response, citation_only)
        if self.speech_output:
            response_as_string = str(response)
            if not response_as_string.startswith("TOOL") and not response_as_string.startswith("{"):
                speak_llm_response(response_as_string)


####### MAIN #######

@click.command()
@click.option('--llm', default="llama3.2", show_default=True, help='model to use from hosted ollama server')
@click.option('--langroid_tools', default=False, is_flag=True, show_default=True, help="optionally, use langroid's tool framework instead of the native functions API")
@click.option('--llm_delegate', default=False, is_flag=True, help="...")
@click.option('--hvac/--no-hvac', default=True, show_default=True, help='whether or not to give the LLM access to the HVAC control')
@click.option('--volume/--no-volume', default=True, show_default=True, help='whether or not to give the LLM access to the IVI audio settings tools')
@click.option('--radio/--no-radio', default=False, show_default=True, help='whether or not to give the LLM access to the internet radio tool')
@click.option('--tts/--no-tts', default=True, show_default=True, help='whether or not to invoke tts to speak output')
@click.option('--input_by_voice/--input_by_text', default=True, show_default=True, help='input by speaking (requires an ASR server to be running) instead of typing')
@click.option('--prompt', type=str, help='filename from which to load custom prompt text')
def main(llm, langroid_tools, llm_delegate, hvac, volume, radio, tts, input_by_voice, prompt):
    # LLM from ollama
    llm_config = lr.language_models.OpenAIGPTConfig(
    chat_model=f"ollama/{llm}", #"ollama/cow/tulu3_tools",
    chat_context_length=16_000, # adjust based on model
    )

    # prompt
    prompt_text = f"""
        You are a driving companion.  Using your tools, you have the capability to:
        {'- find and set radio stations' if radio else ''} 
        {'- adjust the volume' if volume else ''}
        {'- adjust the fan speed of the climate control' if hvac else ''}
        .
        Keep your responses brief and use casual speech.
        It's OK to respond with '[Silence]' if you don't have anything substantial to add to the conversation.
        Do not, under any circumstnances, include an enumerated list in your response.
        By the way, your name is Neptune.
    """
    custom_prompt_path = Path(prompt)
    if custom_prompt_path.exists():
        prompt_text = custom_prompt_path.read_text()

    # langroid agent
    agent_config = lr.ChatAgentConfig(
        name="Neptune",
        llm=llm_config,
        system_message=prompt_text,
        use_tools=langroid_tools,
        use_functions_api=(not langroid_tools),
    )
    agent = CarCompanionAgent(agent_config, voice_input=input_by_voice, speech_output=tts)
    if hvac:
        agent.enable_message(HVACTool)
        print("HVAC tool use enabled")
    if volume:
        agent.enable_message(AudioTool)
        print("Audio tool use enabled")
    if radio:
        agent.enable_message(RadioTool)
        print("Radio tool use enabled")

    # interacton task
    task = lr.Task(agent, interactive=True, llm_delegate=llm_delegate)

    # main loop
    while not agent.user_said_goodbye:
        task.step() 

if __name__ == "__main__":
    main()
