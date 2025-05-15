# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing
the interfacing tasks. In this way you don't mix your true code with
DEEPaaS code and everything is more modular. That is, if you need to write
the predict() function in api.py, you would import your true predict function
and call it from here (with some processing / postprocessing in between
if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at
an exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

from pathlib import Path
import logging
import builtins
from audio_vessel_classifier import config
from audio_vessel_classifier.misc import _catch_error
from transformers import AutoProcessor, ClapModel, ClapAudioModelWithProjection, ClapProcessor
from aiohttp.web import HTTPException
import yaml
import os
import torch
import json
from torch import nn
from collections import OrderedDict
from webargs import fields

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = config.PROJECT_METADATA
        # TODO: Add dynamic metadata collection here
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


@catch_error
def predict(**args):
    logger.debug("Predict with args: %s", args)
    try:
        if not any([args["pt"]]):
            raise Exception(
                "You must provide  '.pt' in the payload"
            )

        return predict_data(args)

    except Exception as err:
        raise HTTPException(reason=err) from err
def return_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Selected CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device('cpu')
    return device

    
def predict_data(args):
    """
    Function to predict from an input tensor in args["file"]
    """
    logger.debug("Predict with args: %s", args)
    try:
        update_with_query_conf(args)
        conf = config.conf_dict

        x = args["file"]
        device = return_device()
        x = x.to(device).squeeze(1)

        # Load models
        clap_model = ClapAudioModelWithProjection.from_pretrained(
            "/srv/DEEP-OC-underwater-noise-classification/models/fine_tuning/model"
        ).to(device)
        
        # Load linear layer (assuming it's a state_dict)
        linear_model =nn.Linear(in_features=512, out_features=11) # adjust dimensions as needed
        linear_model.load_state_dict(torch.load(
            "/srv/DEEP-OC-underwater-noise-classification/models/fine_tuning/model/linear.pth",
            map_location=device
        ))
        linear_model = linear_model.to(device)
        linear_model.eval()

        with torch.no_grad():
            x = clap_model(x).audio_embeds.to(device)
            out = linear_model(x)

        return out


        return out
    except Exception as err:
        raise HTTPException(reason=err) from err

def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                raw_value = user_args[g_key]
                if not raw_value:
                    continue  # skip if the value is empty
                try:
                    # Try parsing as JSON
                    g_val["value"] = json.loads(raw_value)
                except json.JSONDecodeError:
                    # Fall back to treating it as a plain string
                    g_val["value"] = raw_value
    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)



def get_predict_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([("testing", default_conf["testing"])])


    parser["pt"] = fields.Field(
        required=False,
        load_default=None,
        # type="file",
        data_key="pt",
        #  location="form",
        metadata={
            "description": "Select the embedding.",
            "type": "file",
            "location": "form",
        },
    )

    return populate_parser(parser, default_conf)


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    print("popu parser")
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val["help"] if ("help" in gg_keys) else ""
            type = (
                getattr(builtins, g_val["type"])
                if ("type" in gg_keys)
                else None
            )
            choices = (
                g_val["choices"] if ("choices" in gg_keys) else None
            )

            # Additional info in help string
            help += (
                "\n"
                + "<font color='#C5576B'> Group name: **{}**".format(
                    str(group)
                )
            )
            if choices:
                help += "\n" + "Choices: {}".format(str(choices))
            if type:
                help += "\n" + "Type: {}".format(g_val["type"])
            help += "</font>"

            # Create arg dict
            opt_args = {
                "load_default": json.dumps(g_val["value"]),
                "metadata": {"description": help},
                "required": False,
            }
            if choices:
                json_choices = [json.dumps(i) for i in choices]
                opt_args["metadata"]["enum"] = json_choices
                opt_args["validate"] = fields.validate.OneOf(
                    json_choices
                )
            parser[g_key] = fields.Str(**opt_args)

    return parser    
# def warm():
#     pass
#
#
# def get_predict_args():
#     return {}
#
#
# @_catch_error
# def predict(**kwargs):
#     return None
#
#
# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None
