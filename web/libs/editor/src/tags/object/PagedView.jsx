import React, { useCallback, useEffect, useState } from "react";
import { observer } from "mobx-react";
import { types } from "mobx-state-tree";

import Registry from "../../core/Registry";
import Types from "../../core/Types";
import Tree from "../../core/Tree";
import { Pagination } from "../../common/Pagination/Pagination";
import { Hotkey } from "../../core/Hotkey";
import { FF_DEV_1170, isFF } from "../../utils/feature-flags";
import { AnnotationMixin } from "../../mixins/AnnotationMixin";
import Slider, { sliderClasses } from "@mui/joy/Slider";

const Model = types.model({
  id: types.identifier,
  type: "pagedview",
  children: Types.unionArray([
    "view",
    "header",
    "labels",
    "label",
    "table",
    "taxonomy",
    "choices",
    "choice",
    "collapse",
    "datetime",
    "number",
    "rating",
    "ranker",
    "rectangle",
    "ellipse",
    "polygon",
    "keypoint",
    "brush",
    "magicwand",
    "rectanglelabels",
    "ellipselabels",
    "polygonlabels",
    "keypointlabels",
    "brushlabels",
    "hypertextlabels",
    "timeserieslabels",
    "text",
    "audio",
    "image",
    "hypertext",
    "richtext",
    "timeseries",
    "audioplus",
    "list",
    "dialog",
    "textarea",
    "pairwise",
    "style",
    "label",
    "relations",
    "filter",
    "timeseries",
    "timeserieslabels",
    "pagedview",
    "paragraphs",
    "paragraphlabels",
    "video",
    "videorectangle",
  ]),
});

// All attributtes are passed from Repeater tag, noitice attribute name will be lowcased when parsing
const TagAttrs = types.model({
  tooltiplabelkey: types.optional(types.maybeNull(types.string), null),
  highlightannotationmark: types.optional(types.maybeNull(types.boolean), false)
});

const PagedViewModel = types.compose("PagedViewModel", Model, AnnotationMixin, TagAttrs);
const PAGE_QUERY_PARAM = "view_page";
const hotkeys = Hotkey("PagedView");
const DEFAULT_PAGE_SIZE = 1;
const PAGE_SIZE_OPTIONS = [1, 5, 10, 25, 50, 100];

const getStoredPageSize = (name, defaultValue) => {
  const value = localStorage.getItem(`pages:${name}`);

  if (value) {
    return Number.parseInt(value);
  }

  return defaultValue ?? undefined;
};

const setStoredPageSize = (name, pageSize) => {
  localStorage.setItem(`pages:${name}`, pageSize.toString());
};

const getQueryPage = () => {
  const params = new URLSearchParams(window.location.search);
  const page = params.get(PAGE_QUERY_PARAM);

  if (page) {
    return Number.parseInt(page);
  }

  return 1;
};

let lastTaskId = null;

const updateQueryPage = (page, currentTaskId = null) => {
  const params = new URLSearchParams(window.location.search);

  const taskIdChanged = currentTaskId !== lastTaskId;
  const resetPage = lastTaskId && taskIdChanged;

  lastTaskId = currentTaskId;

  if (resetPage) {
    params.delete(PAGE_QUERY_PARAM);
  } else if (page !== 1) {
    params.set(PAGE_QUERY_PARAM, page.toString());
  } else {
    params.delete(PAGE_QUERY_PARAM);
  }

  window.history.replaceState(undefined, undefined, `${window.location.pathname}?${params}`);
};

// custom mark label for MUI joy slider component, take an prop to higlight mark object have annotation (unused)
// const CustomMarkLabel = (props) => {
//   const index = props["data-index"];
//   const mark = props.ownerState.marks[index];

//   return (
//     <span {...props} style={{color}}>
//     </span>
//   );
// };

const HtxPagedView = observer(({ item }) => {
  const [page, _setPage] = useState(getQueryPage);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [marks, setMarks] = useState([]);

  const setPage = useCallback((_page) => {
    _setPage(_page);
    updateQueryPage(_page, item.annotationStore?.store?.task.id);
  }, []);

  const totalPages = Math.ceil(item.children.length / pageSize);
  const sliderLabel = page => {
    let label = `Frame ${page}`;
    try {
      let onKey = item.$treenode._initialSnapshot.on.replace("$", "");
      let labelKey = item.tooltiplabelkey;
      if (labelKey) {
        label =
          item.annotationStore?.store?.task.dataObj[onKey][page - 1][labelKey];
      }
    } catch (e) {
      console.log(e);
    }
    return label;
  };

  useEffect(() => {
    setPageSize(getStoredPageSize("repeater", DEFAULT_PAGE_SIZE));
  }, []);

  useEffect(() => {
    const last = item.annotation.lastSelectedRegion;

    if (last) {
      const _pageNumber = Number.parseFloat(last.object.name.split("_")[1]) + 1;

      setPage(Math.ceil(_pageNumber / pageSize));
    }
  }, [item.annotation.lastSelectedRegion]);

  useEffect(() => {
    if (isFF(FF_DEV_1170)) {
      document.querySelector(".lsf-sidepanels__content")?.scrollTo(0, 0);
    } else {
      document.querySelector("#label-studio-dm")?.scrollTo(0, 0);
    }

    setTimeout(() => {
      hotkeys.addNamed("frame:next-page", () => {
        if (page < totalPages) {
          setPage(page + 1);
        }
      });

      hotkeys.addNamed("frame:previous-page", () => {
        if (page > 1) {
          setPage(page - 1);
        }
      });
    });

    return () => {
      hotkeys.removeNamed("frame:next-page");
      hotkeys.removeNamed("frame:previous-page");
    };
  }, [page]);

  useEffect(() => {
    if (item.highlightannotationmark) {
      const childrenImagesList = item.$treenode._initialSnapshot.children.map(
        (view, index) => {
          let imageName = "";
          view.children.forEach(child => {
            if (child.type === "image") {
              imageName = child.name;
            }
          });
          return {
            name: imageName,
            index: index
          };
        }
      );
      let annotationTargetNameList = item.annotation.results.map(
        item => item.to_name?.name
      );
      const markList = childrenImagesList.map((image, index) => ({
        name: image.name,
        index: index,
        haveAnnotation: annotationTargetNameList.includes(image.name)
      }));
      setMarks(markList);
    }
  }, [item.annotation.results, item.annotation.results.length]);

  useEffect(() => {
    updateQueryPage(getQueryPage(), item.annotationStore?.store?.task.id);
    return () => {
      updateQueryPage(1, item.annotationStore?.store?.task.id);
    };
  }, []);

  const renderPage = useCallback(() => {
    const pageView = [];

    for (let i = 0; i < pageSize; i++) {
      pageView.push(Tree.renderChildren(item.children[i + pageSize * (page - 1)], item.annotation));
    }

    return pageView;
  }, [page, pageSize]);

  return (
    <div>
      <div style={{ width: "100%", padding: '17px 21px 0px 21px', background:'white', position:'sticky', top: 0,zIndex:10, boxShadow:"0px 0px 0px 1px rgba(0, 0, 0, 0.05), 0px 5px 10px rgba(0, 0, 0, 0.1)" }}>
        <Slider
          min={1}
          max={totalPages}
          step={1}
          value={page}
          onChange={(event, value) => {
            item.annotation.unselectAll();
            setPage(value);
          }}
          valueLabelDisplay="on"
          marks={item.highlightannotationmark ? marks.map(mark => ({
            value: mark.index + 1,
            label: mark.haveAnnotation ?"▲" : ""
          })): true}
          valueLabelFormat={value => {
            return sliderLabel(value);
          }}
          sx={{
            "--Slider-markSize": "4px",
            "& .MuiSlider-markLabel": {
              top: 'calc(50% - 6px +(max(var(--Slider-trackSize), var(--Slider-thumbSize)) / 2))',
              color: "#ff5a5a"
            },
            // Need both of the selectors to make it works on the server-side and client-side
            [`& [style*="left:0%"], & [style*="left: 0%"]`]: {
              [`&.${sliderClasses.markLabel}`]: {
                transform: 'none',
              },
              [`& .${sliderClasses.valueLabel}`]: {
                left: 'calc(var(--Slider-thumbSize) / 2)',
                borderBottomLeftRadius: 0,
                '&::before': {
                  left: 0,
                  transform: 'translateY(100%)',
                  borderLeftColor: 'currentColor',
                },
              },
            },
            [`& [style*="left:100%"], & [style*="left: 100%"]`]: {
              [`&.${sliderClasses.markLabel}`]: {
                transform: 'translateX(-100%)',
              },
              [`& .${sliderClasses.valueLabel}`]: {
                right: 'calc(var(--Slider-thumbSize) / 2)',
                borderBottomRightRadius: 0,
                '&::before': {
                  left: 'initial',
                  right: 0,
                  transform: 'translateY(100%)',
                  borderRightColor: 'currentColor',
                },
              },
            },
          }}
          onKeyDown={e => {
            if(e.ctrlKey) {
              e.target.blur();
            }
          
          }}
        ></Slider>
      </div>
      {renderPage()}
    </div>
  );
});

Registry.addTag("pagedview", PagedViewModel, HtxPagedView);

export { HtxPagedView, PagedViewModel };
