// SPDX-License-Identifier: Apache-2.0
//
// la/generate/parse.cpp — see parse.hpp.

#include "la/generate/parse.hpp"

#include <cctype>
#include <cstddef>
#include <regex>

#include "la/generate/coord.hpp"

namespace la::generate {

namespace {

// Classify the coord count inside one box into a DetectionKind.
DetectionKind kind_from_count(std::size_t coord_count) {
    if (coord_count == 4) return DetectionKind::Box;
    if (coord_count == 2) return DetectionKind::Point;
    // Anything else (0, 1, 3, >4) is treated as None geometry; the caller can
    // still inspect coords_permille. The reference FSM only commits 4-coord
    // (coord_box) or 2-coord (point_box) frames, so other counts indicate a
    // malformed/partial box.
    return DetectionKind::None;
}

}  // namespace

ParseResult parse_token_stream(const std::vector<std::int64_t>& ids,
                               const TokenIds& tok,
                               std::vector<RefLabel>* label_ids_out) {
    ParseResult result;

    // Pending label collected from the most recent <ref>...</ref> not yet
    // attached to a box.
    std::optional<RefLabel> pending_ref;

    const std::size_t n = ids.size();
    std::size_t i = 0;
    while (i < n) {
        const std::int64_t id = ids[i];

        if (id == tok.ref_start) {
            RefLabel rl;
            ++i;
            while (i < n && ids[i] != tok.ref_end) {
                rl.ids.push_back(ids[i]);
                ++i;
            }
            if (i < n && ids[i] == tok.ref_end) {
                ++i;  // consume </ref>
            }
            pending_ref = std::move(rl);
            continue;
        }

        if (id == tok.box_start) {
            Detection det;
            RefLabel attached_label;
            if (pending_ref.has_value()) {
                attached_label = *pending_ref;
                pending_ref.reset();
            }

            ++i;  // consume <box>
            // Check the <box>none</box> case first.
            if (i < n && ids[i] == tok.none) {
                det.kind = DetectionKind::None;
                ++i;  // consume none
                if (i < n && ids[i] == tok.box_end) {
                    ++i;  // consume </box>
                }
            } else {
                // Collect coordinate tokens until </box> (or stream end / a new
                // structural token). Non-coord ids inside the box are skipped,
                // matching the FSM which only keeps coord tokens.
                std::vector<std::int64_t> coords;
                while (i < n && ids[i] != tok.box_end &&
                       ids[i] != tok.box_start && ids[i] != tok.ref_start &&
                       ids[i] != tok.im_end) {
                    const std::int64_t cid = ids[i];
                    if (tok.IsCoordToken(cid)) {
                        coords.push_back(tok.CoordBin(cid));  // per-mille
                    } else if (cid == kUncertainCoordId) {
                        coords.push_back(kUncertainCoordId);  // hybrid sentinel
                    }
                    // else: ignore stray non-coord ids.
                    ++i;
                }
                if (i < n && ids[i] == tok.box_end) {
                    ++i;  // consume </box>
                }
                det.kind = kind_from_count(coords.size());
                det.coords_permille = std::move(coords);
            }

            result.detections.push_back(std::move(det));
            if (label_ids_out != nullptr) {
                label_ids_out->push_back(std::move(attached_label));
            }
            continue;
        }

        if (id == tok.im_end) {
            break;  // terminal
        }

        // Any other token does not start a structure; a pending ref not followed
        // by a box is dropped (free-standing ref). Advance.
        ++i;
    }

    return result;
}

ParseResult parse_text(const std::string& text) {
    ParseResult result;

    static const std::regex kRefRe(R"(<ref>(.*?)</ref>)");
    static const std::regex kBoxRe(R"(<box>(.*?)</box>)");
    // A coordinate group is an integer, optionally wrapped in <>. The worker
    // regex emits <int> tokens; bare ints are also accepted.
    static const std::regex kIntRe(R"(<?(-?\d+)>?)");

    auto box_begin = std::sregex_iterator(text.begin(), text.end(), kBoxRe);
    auto box_end = std::sregex_iterator();
    for (auto it = box_begin; it != box_end; ++it) {
        const std::smatch& bm = *it;
        const std::string inner = bm[1].str();

        Detection det;

        // Find a <ref>...</ref> that ends right before this box.
        const std::ptrdiff_t box_start_pos = bm.position(0);
        std::optional<std::string> label;
        for (auto rit = std::sregex_iterator(text.begin(), text.end(), kRefRe);
             rit != std::sregex_iterator(); ++rit) {
            const std::smatch& rm = *rit;
            const std::ptrdiff_t ref_end_pos =
                rm.position(0) + static_cast<std::ptrdiff_t>(rm.length(0));
            if (ref_end_pos <= box_start_pos) {
                label = rm[1].str();  // keep the latest ref before the box
            } else {
                break;
            }
        }
        if (label.has_value()) {
            det.label = label;
        }

        // Trim whitespace from inner to detect the literal "none".
        std::string trimmed = inner;
        while (!trimmed.empty() &&
               std::isspace(static_cast<unsigned char>(trimmed.front()))) {
            trimmed.erase(trimmed.begin());
        }
        while (!trimmed.empty() &&
               std::isspace(static_cast<unsigned char>(trimmed.back()))) {
            trimmed.pop_back();
        }

        if (trimmed == "none") {
            det.kind = DetectionKind::None;
            result.detections.push_back(std::move(det));
            continue;
        }

        std::vector<std::int64_t> coords;
        for (auto cit = std::sregex_iterator(inner.begin(), inner.end(), kIntRe);
             cit != std::sregex_iterator(); ++cit) {
            coords.push_back(
                static_cast<std::int64_t>(std::stoll((*cit)[1].str())));
        }

        if (coords.size() == 4) {
            det.kind = DetectionKind::Box;
        } else if (coords.size() == 2) {
            det.kind = DetectionKind::Point;
        } else {
            det.kind = DetectionKind::None;
        }
        det.coords_permille = std::move(coords);
        result.detections.push_back(std::move(det));
    }

    return result;
}

}  // namespace la::generate
