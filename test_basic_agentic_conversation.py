#!/usr/bin/env python

import re
from pathlib import Path
import subprocess
import socket
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
    r'\s*<\|user\|>.*$'
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
asr_socket.connect((host, port))

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
        f'--port {ivi_port}',
        f'--ip "{ivi_ip}"',
        f'--EventName "{event_name}"'
    ]
    for key, value in kwargs.items():
        command.append(f'--{key} {value}')
    print(f'~~> {" ".join(command)}')
    result = subprocess.call(command, shell=True)
    #result = subprocess.call(command)
    #print(f"command returned {result}")

# LLM from ollama
llm_config = lr.language_models.OpenAIGPTConfig(
    chat_model="ollama/llama3.2", #"ollama/cow/tulu3_tools",
    chat_context_length=16_000, # adjust based on model
)

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
            cls(speed="low"),
            cls(speed="high"),
            (  # ...a tuple of "thought leading to tool", and the tool instance
                "I want to cool the cabin",
                cls(speed="high"),
            ),
            (
                "I want to stop cooling the cabin",
                cls(speed="low"),
            ),
        ]
    
    def handle(self): # -> AgentDoneTool:
        print(f"~> Calling set_fan_speed({self.speed})")
        send_ivi_command("HVACCommand", HVACValue=f'"{self.speed}"')
        return f"Set fan speed to {self.speed}."
        
class AudioTool(lr.ToolMessage):
    request: str = "adjust_audio_volume"
    purpose: str = "To change the volume of the music within the range of 0 (silent) to 10 (maximum volume)"
    volume: int

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        # Examples that will be compiled into few-shot examples for the LLM.
        # Each example can either be...
        return [
            # ... just instances of the tool-class, OR
            cls(volume="4"),
            (  # ...a tuple of "thought leading to tool", and the tool instance
                "I want to crank up the volume",
                cls(volume="7"),
            ),
            (
                "I want to restore the volume to a normal level",
                cls(volume="4"),
            ),
            (
                "I want to lower the volume",
                cls(volume="2")
            )
        ]

    def handle(self): # -> AgentDoneTool:
        print(f"~> Calling adjust_audio_volume({self.volume})")
        send_ivi_command("AudioCommand", Volume=f'"{self.volume}"')
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
        send_ivi_command("RadioCommand", Query=f'"{self.search_term}"')
        return f"Searched for radio station related to {self.search_term} and found it."

####### AGENT + CONVERSATIONAL TASK #######

class CarCompanionAgent(lr.ChatAgent):

    USER_GOODBYE_PHRASE = "goodbye"
    
    def __init__(self, *args, **kwargs) -> None:
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
        response_as_string = str(response)
        if not response_as_string.startswith("TOOL") and not response_as_string.startswith("{"):
            speak_llm_response(response_as_string)

# langroid agent
agent_config = lr.ChatAgentConfig(
    name="Neptune",
    llm=llm_config,
    system_message="""
        You are a driving companion.  Using your tools, you have the capability to (a) find and set radio stations, 
        (b) adjust the volume, and (c) adjust the fan speed of the climate control.
        Keep your responses brief and use casual speech.
        It's OK to respond with '[Silence]' if you don't have anything substantial to add to the conversation.
        Do not, under any circumstnances, include an enumerated list in your response.
        By the way, your name is Neptune.
    """,
    use_tools=True, 
)
agent = CarCompanionAgent(agent_config)
agent.enable_message(HVACTool)
agent.enable_message(AudioTool)
agent.enable_message(RadioTool)

# interacton task
task = lr.Task(agent, interactive=False, llm_delegate=True)

##### MAIN LOOP #####

while not agent.user_said_goodbye:
    task.step() 


