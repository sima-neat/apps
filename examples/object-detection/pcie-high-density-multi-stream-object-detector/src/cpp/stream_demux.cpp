// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stream_demux.h"

#include <gst/gst.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct GstNeatAppStreamDemux {
  GstElement parent;
  GstPad* sink_pad;
  guint source_pad_count;
};

struct GstNeatAppStreamDemuxClass {
  GstElementClass parent_class;
};

G_DEFINE_TYPE(GstNeatAppStreamDemux, gst_neat_app_stream_demux, GST_TYPE_ELEMENT)

GstStaticPadTemplate sink_template =
    GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);
GstStaticPadTemplate source_template =
    GST_STATIC_PAD_TEMPLATE("src_%u", GST_PAD_SRC, GST_PAD_REQUEST, GST_STATIC_CAPS_ANY);

bool parse_index(const char* text, guint* value) {
  if (!text || !*text || !value) {
    return false;
  }
  const char* digits = text;
  if (g_str_has_prefix(text, "stream")) {
    digits += std::strlen("stream");
  }
  if (!*digits) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(digits, &end, 10);
  if (errno != 0 || !end || *end != '\0' || parsed > G_MAXUINT) {
    return false;
  }
  *value = static_cast<guint>(parsed);
  return true;
}

bool structure_stream_id(const GstStructure* structure, const char* field, guint* stream_id) {
  if (!structure || !gst_structure_has_field(structure, field)) {
    return false;
  }
  if (gst_structure_get_uint(structure, field, stream_id)) {
    return true;
  }
  const char* text = gst_structure_get_string(structure, field);
  return parse_index(text, stream_id);
}

bool buffer_stream_id(GstBuffer* buffer, guint* stream_id) {
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  const GstStructure* structure = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  return structure_stream_id(structure, "orig-stream-id", stream_id) ||
         structure_stream_id(structure, "stream-id", stream_id);
}

std::vector<GstPad*> source_pads(GstNeatAppStreamDemux* self) {
  std::vector<GstPad*> pads;
  GST_OBJECT_LOCK(self);
  for (GList* item = GST_ELEMENT(self)->srcpads; item; item = item->next) {
    pads.push_back(GST_PAD(gst_object_ref(item->data)));
  }
  GST_OBJECT_UNLOCK(self);
  return pads;
}

GstFlowReturn chain(GstPad*, GstObject* parent, GstBuffer* buffer) {
  auto* self = reinterpret_cast<GstNeatAppStreamDemux*>(parent);
  guint stream_id = 0;
  if (!buffer_stream_id(buffer, &stream_id)) {
    GST_ELEMENT_ERROR(self, STREAM, FORMAT, ("BBOX result is missing stream-id metadata"), (NULL));
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }

  const std::string pad_name = "src_" + std::to_string(stream_id);
  GstPad* output = gst_element_get_static_pad(GST_ELEMENT(self), pad_name.c_str());
  if (!output) {
    GST_ELEMENT_ERROR(self, STREAM, FORMAT, ("BBOX result targets an unrequested stream"),
                      ("stream=%u", stream_id));
    gst_buffer_unref(buffer);
    return GST_FLOW_NOT_LINKED;
  }
  const GstFlowReturn flow = gst_pad_push(output, buffer);
  gst_object_unref(output);
  return flow;
}

gboolean sink_event(GstPad*, GstObject* parent, GstEvent* event) {
  auto* self = reinterpret_cast<GstNeatAppStreamDemux*>(parent);
  const GstEventType type = GST_EVENT_TYPE(event);
  const auto pads = source_pads(self);
  gboolean accepted = TRUE;

  for (GstPad* output : pads) {
    GstEvent* outgoing = nullptr;
    if (type == GST_EVENT_STREAM_START) {
      const std::string id = "pcie-result-" + std::string(GST_PAD_NAME(output));
      outgoing = gst_event_new_stream_start(id.c_str());
      guint group_id = 0;
      if (gst_event_parse_group_id(event, &group_id)) {
        gst_event_set_group_id(outgoing, group_id);
      }
    } else {
      outgoing = gst_event_ref(event);
    }
    accepted = gst_pad_push_event(output, outgoing) && accepted;
    gst_object_unref(output);
  }
  gst_event_unref(event);
  return accepted;
}

GstPad* request_new_pad(GstElement* element, GstPadTemplate* templ, const gchar* requested_name,
                        const GstCaps*) {
  auto* self = reinterpret_cast<GstNeatAppStreamDemux*>(element);
  if (GST_PAD_TEMPLATE_DIRECTION(templ) != GST_PAD_SRC) {
    return nullptr;
  }

  guint stream_id = self->source_pad_count;
  if (requested_name && g_strcmp0(requested_name, "src_%u") != 0) {
    if (!g_str_has_prefix(requested_name, "src_") ||
        !parse_index(requested_name + std::strlen("src_"), &stream_id)) {
      GST_ERROR_OBJECT(self, "invalid source pad name: %s", requested_name);
      return nullptr;
    }
  }
  if (stream_id != self->source_pad_count) {
    GST_ERROR_OBJECT(self, "stream pads must be contiguous: expected src_%u, got src_%u",
                     self->source_pad_count, stream_id);
    return nullptr;
  }

  const std::string name = "src_" + std::to_string(stream_id);
  GstPad* pad = gst_pad_new_from_template(templ, name.c_str());
  if (!pad || !gst_element_add_pad(element, pad)) {
    if (pad) {
      gst_object_unref(pad);
    }
    return nullptr;
  }
  ++self->source_pad_count;
  return pad;
}

void release_pad(GstElement* element, GstPad* pad) {
  auto* self = reinterpret_cast<GstNeatAppStreamDemux*>(element);
  gst_element_remove_pad(element, pad);
  GST_OBJECT_LOCK(self);
  self->source_pad_count = static_cast<guint>(g_list_length(GST_ELEMENT(self)->srcpads));
  GST_OBJECT_UNLOCK(self);
}

void gst_neat_app_stream_demux_class_init(GstNeatAppStreamDemuxClass* klass) {
  auto* element_class = GST_ELEMENT_CLASS(klass);
  element_class->request_new_pad = request_new_pad;
  element_class->release_pad = release_pad;
  gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&sink_template));
  gst_element_class_add_pad_template(element_class, gst_static_pad_template_get(&source_template));
  gst_element_class_set_static_metadata(
      element_class, "NEAT Apps stream result demultiplexer", "Demux",
      "Routes correlated model results to numbered source pads without copying", "SiMa.ai NEAT");
}

void gst_neat_app_stream_demux_init(GstNeatAppStreamDemux* self) {
  self->source_pad_count = 0;
  self->sink_pad = gst_pad_new_from_static_template(&sink_template, "sink");
  gst_pad_set_chain_function(self->sink_pad, GST_DEBUG_FUNCPTR(chain));
  gst_pad_set_event_function(self->sink_pad, GST_DEBUG_FUNCPTR(sink_event));
  gst_element_add_pad(GST_ELEMENT(self), self->sink_pad);
}

} // namespace

namespace pcie_high_density {

bool register_stream_demux() {
  return gst_element_register(nullptr, "neatappstreamdemux", GST_RANK_NONE,
                              gst_neat_app_stream_demux_get_type()) == TRUE;
}

} // namespace pcie_high_density
