import os
from typing import List
from typing import Tuple
import logging
from collections import defaultdict
from collections import Counter
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random 

import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', 'Reason-SCAN', 'code', 'dataset'))

# we want to make this file device irrelevant,
# and we decide where to store the data afterwards.
# def isnotebook():
#     try:
#         shell = get_ipython().__class__.__name__
#         if shell == 'ZMQInteractiveShell':
#             return True   # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False      # Probably standard Python interpreter
# if isnotebook():
#     device = torch.device("cpu")
# else:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)

from world import *
from vocabulary import Vocabulary as ReaSCANVocabulary
from object_vocabulary import *

class Vocabulary(object):
    """
    Object that maps words in string form to indices to be processed by numerical models.
    """

    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def sos_idx(self):
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self):
        return self.word_to_idx(self.eos_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path
    
class ReaSCANDataset(object):
    """
    Loads a GroundedScan instance from a specified location.
    """
    def __init__(self, data_json, save_directory: str, k: int, split="all", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False):
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, input_vocabulary_file)) and os.path.exists(
                os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        
        # we simply load the json file.
        logger.info(f"Formulating the dataset from the passed in json file...")
        self.data_json = data_json
        
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
            
        # some helper initialization
        self.grid_size = self.data_json['grid_size']

        intransitive_verbs = ["walk"]
        transitive_verbs = ["push", "pull"]
        adverbs = ["while zigzagging", "while spinning", "cautiously", "hesitantly"]
        nouns = ["circle", "cylinder", "square", "box"]
        color_adjectives = ["red", "blue", "green", "yellow"]
        size_adjectives = ["big", "small"]
        relative_pronouns = ["that is"]
        relation_clauses = ["in the same row as", 
                            "in the same column as", 
                            "in the same color as", 
                            "in the same shape as", 
                            "in the same size as",
                            "inside of"]
        reaSCANVocabulary = ReaSCANVocabulary.initialize(intransitive_verbs=intransitive_verbs,
                                                   transitive_verbs=transitive_verbs, adverbs=adverbs, nouns=nouns,
                                                   color_adjectives=color_adjectives,
                                                   size_adjectives=size_adjectives, 
                                                   relative_pronouns=relative_pronouns, 
                                                   relation_clauses=relation_clauses)
        min_object_size = 1
        max_object_size = 4
        object_vocabulary = ObjectVocabulary(shapes=reaSCANVocabulary.get_semantic_shapes(),
                                             colors=reaSCANVocabulary.get_semantic_colors(),
                                             min_size=min_object_size, max_size=max_object_size)
        
        self._world = World(grid_size=self.grid_size, colors=reaSCANVocabulary.get_semantic_colors(),
                            object_vocabulary=object_vocabulary,
                            shapes=reaSCANVocabulary.get_semantic_shapes(),
                            save_directory=save_directory)
        self._world.clear_situation()
            
        self.image_dimensions = self._world.get_current_situation_image().shape[0] 
        self.image_channels = 3
        self.split = split
        self.directory = save_directory
        self.k = k

        # Keeping track of data.
        self._examples = np.array([])
        self._input_lengths = np.array([])
        self._target_lengths = np.array([])
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")
    
        self.k=k # this is for the few-shot splits only!

    @staticmethod
    def command_repr(command: List[str]) -> str:
        return ','.join(command)

    @staticmethod
    def parse_command_repr(command_repr: str) -> List[str]:
        return command_repr.split(',')
    
    def initialize_world(self, situation: Situation, mission="") -> {}:
        """
        Initializes the world with the passed situation.
        :param situation: class describing the current situation in the world, fully determined by a grid size,
        agent position, agent direction, list of placed objects, an optional target object and optional carrying object.
        :param mission: a string defining a command (e.g. "Walk to a green circle.")
        """
        objects = []
        for positioned_object in situation.placed_objects:
            objects.append((positioned_object.object, positioned_object.position))
        self._world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                               target_object=situation.target_object, carrying=situation.carrying)
        if mission:
            self._world.set_mission(mission)

    def extract_size_color_shape(self, referred_target_str):
        target_size_d = ""
        target_color_d = ""
        target_shape_d = ""
        if "small" in referred_target_str:
            target_size_d = "small"
        elif "big" in referred_target_str:
            target_size_d = "big"
        else:
            pass
        if "circle" in referred_target_str:
            target_shape_d = "circle"
        elif "cylinder" in referred_target_str:
            target_shape_d = "cylinder"
        elif "square" in referred_target_str:
            target_shape_d = "square"
        elif "box" in referred_target_str:
            assert False
        elif "object" in referred_target_str:
            assert False
        else:
            pass
        if "red" in referred_target_str:
            target_color_d = "red"
        elif "blue" in referred_target_str:
            target_color_d = "blue"
        elif "green" in referred_target_str:
            target_color_d = "green"
        elif "yellow" in referred_target_str:
            target_color_d = "yellow"
        else:
            pass
        
        return target_size_d, target_color_d, target_shape_d
    
    def get_examples_with_image(self, split="train", simple_situation_representation=False) -> dict:
        """
        Get data pairs with images in the form of np.ndarray's with RGB values or with 1 pixel per grid cell
        (see encode in class Grid of minigrid.py for details on what such representation looks like).
        :param split: string specifying which split to load.
        :param simple_situation_representation:  whether to get the full RGB image or a simple representation.
        :return: data examples.
        """
        for example in self.data_json["examples"][split]:
            target_size_d, target_color_d, target_shape_d = self.extract_size_color_shape(example["referred_target"])
            target_serialized_str = ",".join([target_size_d, target_color_d, target_shape_d])

            command = self.parse_command_repr(example["command"])
            if example.get("meaning"):
                meaning = example["meaning"]
            else:
                meaning = example["command"]
            meaning = self.parse_command_repr(meaning)
            situation = Situation.from_representation(example["situation"])
            self._world.clear_situation()
            self.initialize_world(situation)
            if simple_situation_representation:
                situation_image = self._world.get_current_situation_grid_repr()
            else:
                situation_image = self._world.get_current_situation_image()
            target_commands = self.parse_command_repr(example["target_commands"])
            verb_in_command = ""
            if "verb_in_command" in example:
                verb_in_command = example["verb_in_command"]
            adverb_in_command = ""
            if "adverb_in_command" in example:
                adverb_in_command = example["adverb_in_command"]
            yield {"input_command": command, "input_meaning": meaning,
                   "derivation_representation": example.get("derivation"),
                   "situation_image": situation_image, "situation_representation": example["situation"],
                   "target_command": target_commands, "target_str": target_serialized_str, 
                   "verb_in_command": verb_in_command, 
                   "adverb_in_command": adverb_in_command}
    
    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.get_examples_with_image(self.split)):
            self.input_vocabulary.add_sentence(example["input_command"])
            self.target_vocabulary.add_sentence(example["target_command"])

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str):
        self.input_vocabulary.save(os.path.join(self.directory, input_vocabulary_file))
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self.input_vocabulary
        elif vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def shuffle_data(self) -> {}:
        """
        Reorder the data examples and reorder the lengths of the input and target commands accordingly.
        """
        random_permutation = np.random.permutation(len(self._examples))
        self._examples = self._examples[random_permutation]
        self._target_lengths = self._target_lengths[random_permutation]
        self._input_lengths = self._input_lengths[random_permutation]
    
    def get_max_seq_length_input(self):
        input_lengths = self._input_lengths
        max_input_length = np.max(input_lengths)
        return max_input_length
     
    def get_max_seq_length_target(self):
        target_lengths = self._target_lengths
        max_target_length = np.max(target_lengths)
        return max_target_length

    def get_dual_dataset(self, novel_attribute=False):
        """
        Function for getting dual dataset for
        counterfactual training.
        """
        examples = self._examples        
        # get length.
        input_lengths = self._input_lengths
        target_lengths = self._target_lengths
        
        ## WARNING: DO NOT UNCOMMENT THE FOLLOWING LINES.
        ## TODO: DEBUG ON THIS.
        # we may want to random shuffle the examples
        # to make sure have a richer space of composities.
        # p = np.random.permutation(len(input_lengths))
        # examples = examples[p]
        # input_lengths = input_lengths[p]
        # target_lengths = target_lengths[p]
        
        max_input_length = np.max(input_lengths)
        max_target_length = np.max(target_lengths)
        
        # return structs
        input_batch = []
        target_batch = []
        situation_batch = []
        situation_representation_batch = []
        derivation_representation_batch = []
        agent_positions_batch = []
        target_positions_batch = []
        target_str_batch = [] 
        adverb_str_batch = []
        verb_str_batch = []
        
        for example in examples:
            to_pad_input = max_input_length - example["input_tensor"].size(1)
            to_pad_target = max_target_length - example["target_tensor"].size(1)
            padded_input = torch.cat([
                example["input_tensor"],
                torch.zeros(int(to_pad_input), dtype=torch.long).unsqueeze(0)], dim=1)
            # padded_input = torch.cat([
            #     torch.zeros_like(example["input_tensor"], dtype=torch.long),
            #     torch.zeros(int(to_pad_input), dtype=torch.long).unsqueeze(0)], dim=1) # TODO: change back
            padded_target = torch.cat([
                example["target_tensor"],
                torch.zeros(int(to_pad_target), dtype=torch.long).unsqueeze(0)], dim=1)
            input_batch.append(padded_input)
            target_batch.append(padded_target)
            situation_batch.append(example["situation_tensor"])
            situation_representation_batch.append(example["situation_representation"])
            derivation_representation_batch.append(example["derivation_representation"])
            agent_positions_batch.append(example["agent_position"])
            target_positions_batch.append(example["target_position"])
            target_str_batch.append(example["target_str"])
            adverb_str_batch.append(example["adverb_in_command"])
            verb_str_batch.append(example["verb_in_command"])
        
        # Main dataset.
        main_input_batch = torch.cat(input_batch, dim=0)
        main_target_batch = torch.cat(target_batch, dim=0)
        main_situation_batch = torch.cat(situation_batch, dim=0)
        main_agent_positions_batch = torch.cat(agent_positions_batch, dim=0)
        main_target_positions_batch = torch.cat(target_positions_batch, dim=0)
        main_input_lengths_batch = torch.tensor([[l] for l in input_lengths], dtype=torch.long)
        main_target_lengths_batch = torch.tensor([[l] for l in target_lengths], dtype=torch.long)
        
        # Dual dataset for counterfactual training.
        dual_input_batch = torch.cat(input_batch, dim=0)
        dual_target_batch = torch.cat(target_batch, dim=0)
        dual_situation_batch = torch.cat(situation_batch, dim=0)
        dual_agent_positions_batch = torch.cat(agent_positions_batch, dim=0)
        dual_target_positions_batch = torch.cat(target_positions_batch, dim=0)
        dual_input_lengths_batch = torch.tensor([[l] for l in input_lengths], dtype=torch.long)
        dual_target_lengths_batch = torch.tensor([[l] for l in target_lengths], dtype=torch.long)
        # Randomly shifting the dataset.
        # Later, we may need to have more complicated sampling strategies for
        # getting the dual dataset for counterfactual training.
        perm_idx = torch.randperm(dual_input_batch.size()[0])
        dual_input_batch = dual_input_batch.index_select(dim=0, index=perm_idx)
        dual_target_batch = dual_target_batch.index_select(dim=0, index=perm_idx)
        dual_situation_batch = dual_situation_batch.index_select(dim=0, index=perm_idx)
        dual_agent_positions_batch = dual_agent_positions_batch.index_select(dim=0, index=perm_idx)
        dual_target_positions_batch = dual_target_positions_batch.index_select(dim=0, index=perm_idx)
        dual_input_lengths_batch = dual_input_lengths_batch.index_select(dim=0, index=perm_idx)
        dual_target_lengths_batch = dual_target_lengths_batch.index_select(dim=0, index=perm_idx)
        dual_target_str_batch = [target_str_batch[i] for i in perm_idx.tolist()]
        dual_situation_representation_batch = [situation_representation_batch[i] for i in perm_idx.tolist()]
        dual_adverb_str_batch = [adverb_str_batch[i] for i in perm_idx.tolist()]
        dual_verb_str_batch = [verb_str_batch[i] for i in perm_idx.tolist()]
        # we need to do a little extra work here just to generate
        # examples for novel attribute cases.
        
        if not novel_attribute:
            main_dataset = TensorDataset(
                # main dataset
                main_input_batch, main_target_batch, main_situation_batch, main_agent_positions_batch, 
                main_target_positions_batch, main_input_lengths_batch, main_target_lengths_batch,
                # dual dataset
                dual_input_batch, dual_target_batch, dual_situation_batch, dual_agent_positions_batch,
                dual_target_positions_batch, dual_input_lengths_batch, dual_target_lengths_batch,
            )
            # with non-tensorized outputs
            return main_dataset, (situation_representation_batch, derivation_representation_batch)
            # the last two items are deprecated. we need to fix them to make them usable.
        
        # here are the steps:
        # 1. find avaliable attributes to swap in both example.
        # 2. swap attribute, and get the updated action sequence, everything else stays the same.
        # 3. we need to identify the swap index.
        # 3.1. there can be two ways of swapping, either whole token swapping, or noun embedding
        # slice swapping. Both should be provided I think.
        # I think we then need to return two things, noun swapping index, and atribute swapping index
        # those two could be the same, if we are swapping the noun.
        batch_size = dual_input_batch.shape[0]
        intervened_main_swap_index = []
        intervened_dual_swap_index = []
        intervened_main_shape_index = []
        intervened_dual_shape_index = []
        intervened_target_batch = []
        intervened_target_lengths_batch = []
        intervened_swap_attr = [] # 0, 1, 2 maps to size, color, shape
        intervened_target_str = []
        for i in range(0, batch_size):
            if not novel_attribute:
                # we put dummies
                main_swap_index, dual_swap_index, main_shape_index, dual_shape_index = -1, -1, -1, -1
                to_pad_target = max_target_length
                intervened_padded_target = torch.zeros(int(to_pad_target), dtype=torch.long)
                intervened_main_swap_index += [main_swap_index]
                intervened_dual_swap_index += [dual_swap_index]
                intervened_main_shape_index += [main_shape_index]
                intervened_dual_shape_index += [dual_shape_index]
                intervened_target_batch += [intervened_padded_target]
                intervened_target_lengths_batch += [0]
                intervened_target_str += [""]
                continue
            main_command_str = self.array_to_sentence(main_input_batch[i].tolist(), "input")
            dual_command_str = self.array_to_sentence(dual_input_batch[i].tolist(), "input")
            target_si, target_co, target_sh = target_str_batch[i].split(",")
            dual_target_si, dual_target_co, dual_target_sh = dual_target_str_batch[i].split(",")
            potential_swap_attr = []
            if target_si != "" and dual_target_si != "" and target_si == dual_target_si:
                potential_swap_attr += ["size"]
            if target_co != "" and dual_target_co != "" and target_co == dual_target_co:
                potential_swap_attr += ["color"]
            if target_sh != "" and dual_target_sh != "" and target_sh == dual_target_sh:
                potential_swap_attr += ["shape"]
            swap_attr = random.choice(potential_swap_attr)
            
            if swap_attr == "size":
                swap_attr_main = target_si
                swap_attr_dual = dual_target_si
                new_composites = [dual_target_si, target_co, target_sh]
                intervened_swap_attr += [0]
            elif swap_attr == "color":
                swap_attr_main = target_co
                swap_attr_dual = dual_target_co
                new_composites = [target_si, dual_target_co, target_sh]
                intervened_swap_attr += [1]
            elif swap_attr == "shape":
                swap_attr_main = target_sh
                swap_attr_dual = dual_target_sh
                new_composites = [target_si, target_co, dual_target_sh]
                intervened_swap_attr += [2]
            
            new_target_id = -1
            id_size_tuples = []
            for k, v in situation_representation_batch[i]["placed_objects"].items():
                if v["object"]["shape"] == new_composites[2]:
                    if new_composites[1] != "":
                        if v["object"]["color"] == new_composites[1]:
                            id_size_tuples.append((k, int(v["object"]["size"])))
                    else:
                        id_size_tuples.append((k, int(v["object"]["size"])))

            if target_si != "":
                # we need to ground size relatively?
                if len(id_size_tuples) == 2:
                    id_size_tuples = sorted(id_size_tuples, key=lambda x: x[1])
                    # only more than 2 we can have relative stuffs.
                    if target_si == "big":
                        new_target_id = id_size_tuples[-1][0]
                    elif target_si == "small":
                        new_target_id = id_size_tuples[0][0]
            else:
                if len(id_size_tuples) == 1:
                    new_target_id = id_size_tuples[0][0]
            
            if new_target_id == -1:
                # we don't have a new target, we need to use some dummy data!
                main_swap_index, dual_swap_index, main_shape_index, dual_shape_index = -1, -1, -1, -1
                to_pad_target = max_target_length
                intervened_padded_target = torch.zeros(int(to_pad_target), dtype=torch.long)
                intervened_target_lengths_batch += [0]
                intervened_target_str += [""]
            else:
                new_target_shape = situation_representation_batch[i]["placed_objects"][new_target_id]['object']['shape']
                new_target_color = situation_representation_batch[i]["placed_objects"][new_target_id]['object']['color']
                assert new_target_shape == new_composites[2]
                if new_composites[1] != "":
                    assert new_target_color == new_composites[1]

                # we have a new target, let us generate the action sequence as well.
                new_target_pos = situation_representation_batch[i]["placed_objects"][new_target_id]["position"]
                self._world.clear_situation()
                for obj_idx, obj in situation_representation_batch[i]["placed_objects"].items():
                    self._world.place_object(
                        Object(size=int(obj["object"]["size"]), color=obj["object"]["color"], shape=obj["object"]["shape"]),
                        position=Position(row=int(obj["position"]["row"]), column=int(obj["position"]["column"]))
                    )
                self._world.place_agent_at(
                    Position(
                        row=int(situation_representation_batch[i]["agent_position"]["row"]),
                        column=int(situation_representation_batch[i]["agent_position"]["column"])
                ))
                row = int(new_target_pos["row"])
                column = int(new_target_pos["column"])
                new_target_position = Position(
                    row=row,
                    column=column
                )
                self._world.go_to_position(
                    position=new_target_position, 
                    manner=adverb_str_batch[i], 
                    primitive_command="walk"
                )

                if len(adverb_str_batch[i].split(" ")) > 1:
                    assert adverb_str_batch[i].split(" ")[-1] in main_command_str
                elif len(adverb_str_batch[i]) != 0:
                    assert adverb_str_batch[i] in main_command_str

                if verb_str_batch[i] != "walk":
                    self._world.move_object_to_wall(action=verb_str_batch[i], manner=adverb_str_batch[i])
                target_commands, _ = self._world.get_current_observations()
                target_array = self.sentence_to_array(target_commands, vocabulary="target")
                self._world.clear_situation()
                
                # now, we need to get the index of words in the sequence that need to be swapped.
                main_swap_index = main_command_str.index(swap_attr_main)
                dual_swap_index = dual_command_str.index(swap_attr_dual)
                main_shape_index = main_command_str.index(target_sh)
                dual_shape_index = dual_command_str.index(dual_target_sh)
                
                assert main_command_str[main_shape_index] in ["circle", "square", "cylinder"]
                assert dual_command_str[dual_shape_index] in ["circle", "square", "cylinder"]
                if intervened_swap_attr[-1] == 0:
                    assert main_command_str[main_swap_index] in ["", "small", "big"]
                    assert dual_command_str[dual_swap_index] in ["", "small", "big"]
                elif intervened_swap_attr[-1] == 1:
                    assert main_command_str[main_swap_index] in ["", "red", "blue", "green", "yellow"]
                    assert dual_command_str[dual_swap_index] in ["", "red", "blue", "green", "yellow"]
                elif intervened_swap_attr[-1] == 2:
                    assert main_command_str[main_swap_index] in ["circle", "cylinder", "square", "box"]
                    assert dual_command_str[dual_swap_index] in ["circle", "cylinder", "square", "box"]
                
                if len(target_array) <= max_target_length:
                    # only these are valid!
                    target_array = torch.tensor(target_array, dtype=torch.long)
                    to_pad_target = max_target_length - target_array.size(0)
                    intervened_target_lengths_batch += [target_array.size(0)]
                    intervened_padded_target = torch.cat([
                        target_array,
                        torch.zeros(int(to_pad_target), dtype=torch.long)], dim=-1)
                    intervened_target_str += [",".join(new_composites)]
                    print(dual_target_str_batch[i])
                    print(target_str_batch[i])
                    print(",".join(new_composites))
                    print(swap_attr)
                else:
                    # we don't have a valid action sequence, we need to use some dummy data!
                    main_swap_index, dual_swap_index, main_shape_index, dual_shape_index = -1, -1, -1, -1
                    to_pad_target = max_target_length
                    intervened_padded_target = torch.zeros(int(to_pad_target), dtype=torch.long)
                    intervened_target_lengths_batch += [0]
            
            # we now consolidate everything.
            intervened_main_swap_index += [main_swap_index]
            intervened_dual_swap_index += [dual_swap_index]
            intervened_main_shape_index += [main_shape_index]
            intervened_dual_shape_index += [dual_shape_index]
            intervened_target_batch += [intervened_padded_target]
        
        intervened_main_swap_index = torch.tensor(intervened_main_swap_index, dtype=torch.long)
        intervened_dual_swap_index = torch.tensor(intervened_dual_swap_index, dtype=torch.long)
        intervened_main_shape_index = torch.tensor(intervened_main_shape_index, dtype=torch.long)
        intervened_dual_shape_index = torch.tensor(intervened_dual_shape_index, dtype=torch.long)
        intervened_target_batch = torch.stack(intervened_target_batch, dim=0)
        intervened_swap_attr = torch.tensor(intervened_swap_attr, dtype=torch.long)
        intervened_target_lengths_batch = torch.tensor(intervened_target_lengths_batch, dtype=torch.long)
        
        assert novel_attribute == True
        main_dataset = TensorDataset(
            # main dataset
            main_input_batch, main_target_batch, main_situation_batch, main_agent_positions_batch, 
            main_target_positions_batch, main_input_lengths_batch, main_target_lengths_batch,
            # dual dataset
            dual_input_batch, dual_target_batch, dual_situation_batch, dual_agent_positions_batch,
            dual_target_positions_batch, dual_input_lengths_batch, dual_target_lengths_batch,
            # intervened dataset for novel attribute
            intervened_main_swap_index, intervened_dual_swap_index, intervened_main_shape_index, 
            intervened_dual_shape_index, intervened_target_batch, intervened_swap_attr, 
            intervened_target_lengths_batch
        )  
        # with non-tensorized outputs
        return main_dataset, (situation_representation_batch, derivation_representation_batch)
        # the last two items are deprecated. we need to fix them to make them usable.
    
    def get_dataset(self):
        examples = self._examples
        
        # get length.
        input_lengths = self._input_lengths
        target_lengths = self._target_lengths
        max_input_length = np.max(input_lengths)
        max_target_length = np.max(target_lengths)
        
        # return structs
        input_batch = []
        target_batch = []
        situation_batch = []
        situation_representation_batch = []
        derivation_representation_batch = []
        agent_positions_batch = []
        target_positions_batch = []

        for example in examples:
            to_pad_input = max_input_length - example["input_tensor"].size(1)
            to_pad_target = max_target_length - example["target_tensor"].size(1)
            padded_input = torch.cat([
                example["input_tensor"],
                torch.zeros(int(to_pad_input), dtype=torch.long).unsqueeze(0)], dim=1)
            # padded_input = torch.cat([
            #     torch.zeros_like(example["input_tensor"], dtype=torch.long),
            #     torch.zeros(int(to_pad_input), dtype=torch.long).unsqueeze(0)], dim=1) # TODO: change back
            padded_target = torch.cat([
                example["target_tensor"],
                torch.zeros(int(to_pad_target), dtype=torch.long).unsqueeze(0)], dim=1)
            input_batch.append(padded_input)
            target_batch.append(padded_target)
            situation_batch.append(example["situation_tensor"])
            situation_representation_batch.append(example["situation_representation"])
            derivation_representation_batch.append(example["derivation_representation"])
            agent_positions_batch.append(example["agent_position"])
            target_positions_batch.append(example["target_position"])
        
        input_batch = torch.cat(input_batch, dim=0)
        target_batch = torch.cat(target_batch, dim=0)
        situation_batch = torch.cat(situation_batch, dim=0)
        agent_positions_batch = torch.cat(agent_positions_batch, dim=0)
        target_positions_batch = torch.cat(target_positions_batch, dim=0)
        input_lengths_batch = torch.tensor([[l] for l in input_lengths], dtype=torch.long)
        target_lengths_batch = torch.tensor([[l] for l in target_lengths], dtype=torch.long)
        main_dataset = TensorDataset(
            input_batch, target_batch, situation_batch, agent_positions_batch, 
            target_positions_batch, input_lengths_batch, target_lengths_batch
        )

        # with non-tensorized outputs
        return main_dataset, (situation_representation_batch, derivation_representation_batch)
        
            
    """
    Deprecated. We may want to deprecate this function, so we want use multi-gpu settings.
    """
    def get_data_iterator(self, batch_size=10) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[dict],
                                                        torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Iterate over batches of example tensors, pad them to the max length in the batch and yield.
        :param batch_size: how many examples to put in each batch.
        :param auxiliary_task: if true, also batches agent and target positions (flattened, so
        agent row * agent columns = agent_position)
        :return: tuple of input commands batch, corresponding input lengths, situation image batch,
        list of corresponding situation representations, target commands batch and corresponding target lengths.
        """
        for example_i in range(0, len(self._examples), batch_size):
            if example_i + batch_size > len(self._examples):
                batch_size = len(self._examples) - example_i
            examples = self._examples[example_i:example_i + batch_size]
            input_lengths = self._input_lengths[example_i:example_i + batch_size]
            target_lengths = self._target_lengths[example_i:example_i + batch_size]
            max_input_length = np.max(input_lengths)
            max_target_length = np.max(target_lengths)
            input_batch = []
            target_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            for example in examples:
                to_pad_input = max_input_length - example["input_tensor"].size(1)
                to_pad_target = max_target_length - example["target_tensor"].size(1)
                padded_input = torch.cat([
                    example["input_tensor"],
                    torch.zeros(int(to_pad_input), dtype=torch.long).unsqueeze(0)], dim=1)
                # padded_input = torch.cat([
                #     torch.zeros_like(example["input_tensor"], dtype=torch.long),
                #     torch.zeros(int(to_pad_input), dtype=torch.long).unsqueeze(0)], dim=1) # TODO: change back
                padded_target = torch.cat([
                    example["target_tensor"],
                    torch.zeros(int(to_pad_target), dtype=torch.long).unsqueeze(0)], dim=1)
                input_batch.append(padded_input)
                target_batch.append(padded_target)
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

            yield (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
                   torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
                   target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

    def read_dataset(self, max_examples=None, simple_situation_representation=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        """
        few_shots_ids = []
        logger.info("Converting dataset to tensors...")
        if self.split == "few_shot_single_clause_logic" and self.k != 0:
            logger.info("Removing examples for few-shots training test set...")
            path_to_few_shot_data = os.path.join(self.directory, f"few-shot-inoculations-{self.k}.txt")
            logger.info(f"Reading few-shot inoculation from file: {path_to_few_shot_data}...")
            few_shots_ids = json.load(open(path_to_few_shot_data, "r"))
        
        for i, example in enumerate(self.get_examples_with_image(self.split, simple_situation_representation)):
            if i in few_shots_ids: # this is just for few-shot experiments.
                continue
            if max_examples:
                if len(self._examples) > max_examples - 1:
                    break
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            #equivalent_target_commands = example["equivalent_target_command"]
            situation_image = example["situation_image"]
            if i == 0:
                self.image_dimensions = situation_image.shape[0]
                self.image_channels = situation_image.shape[-1]
            situation_repr = example["situation_representation"]
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long).unsqueeze(
                dim=0)
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long).unsqueeze(
                dim=0)
            #empty_example["equivalent_target_tensor"] = torch.tensor(equivalent_target_array, dtype=torch.long).unsqueeze(dim=0)
            empty_example["situation_tensor"] = torch.tensor(situation_image, dtype=torch.float).unsqueeze(dim=0)
            empty_example["situation_representation"] = situation_repr
            empty_example["derivation_representation"] = example["derivation_representation"]
            empty_example["agent_position"] = torch.tensor(
                (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["agent_position"]["column"]), dtype=torch.long).unsqueeze(dim=0)
            empty_example["target_position"] = torch.tensor(
                (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["target_object"]["position"]["column"]),
                dtype=torch.long).unsqueeze(dim=0)
            empty_example["target_str"] = example["target_str"]
            empty_example["adverb_in_command"] = example["adverb_in_command"]
            empty_example["verb_in_command"] = example["verb_in_command"]
            self._input_lengths = np.append(self._input_lengths, [len(input_array)])
            self._target_lengths = np.append(self._target_lengths, [len(target_array)])
            self._examples = np.append(self._examples, [empty_example])
        
        # we also need to load few-shots examples in case k is not 0.
        if self.k != 0:
            logger.info("Loading few examples into the training set for few-shots learning...")
            # Let us also record the few shots examples index, so in evaluation,
            # we can move them out!
            few_shot_single_clause_logic = self.data_json["examples"]["few_shot_single_clause_logic"]
            few_shots_ids = [i for i in range(len(few_shot_single_clause_logic))]
            few_shots_ids = random.sample(few_shots_ids, self.k)
            logger.info("The following idx examples are selected for few-shot learning:")
            logger.info(few_shots_ids)
            with open(os.path.join(self.directory, f"few-shot-inoculations-{self.k}.txt"), "w") as fd:
                json.dump(few_shots_ids, fd, indent=4)
                
            all_examples_few_shots_selected = []
            for i, example in enumerate(
                self.get_examples_with_image(
                    "few_shot_single_clause_logic", simple_situation_representation
                )
            ):
                if i in few_shots_ids:
                    all_examples_few_shots_selected.append(example)
            for i, example in enumerate(all_examples_few_shots_selected):
                empty_example = {}
                input_commands = example["input_command"]
                target_commands = example["target_command"]
                #equivalent_target_commands = example["equivalent_target_command"]
                situation_image = example["situation_image"]
                if i == 0:
                    self.image_dimensions = situation_image.shape[0]
                    self.image_channels = situation_image.shape[-1]
                situation_repr = example["situation_representation"]
                input_array = self.sentence_to_array(input_commands, vocabulary="input")
                target_array = self.sentence_to_array(target_commands, vocabulary="target")
                #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
                empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long).unsqueeze(
                    dim=0)
                empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long).unsqueeze(
                    dim=0)
                #empty_example["equivalent_target_tensor"] = torch.tensor(equivalent_target_array, dtype=torch.long).unsqueeze(dim=0)
                empty_example["situation_tensor"] = torch.tensor(situation_image, dtype=torch.float).unsqueeze(dim=0)
                empty_example["situation_representation"] = situation_repr
                empty_example["derivation_representation"] = example["derivation_representation"]
                empty_example["agent_position"] = torch.tensor(
                    (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                    int(situation_repr["agent_position"]["column"]), dtype=torch.long).unsqueeze(dim=0)
                empty_example["target_position"] = torch.tensor(
                    (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                    int(situation_repr["target_object"]["position"]["column"]),
                    dtype=torch.long).unsqueeze(dim=0)
                self._input_lengths = np.append(self._input_lengths, [len(input_array)])
                self._target_lengths = np.append(self._target_lengths, [len(target_array)])
                self._examples = np.append(self._examples, [empty_example])

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size