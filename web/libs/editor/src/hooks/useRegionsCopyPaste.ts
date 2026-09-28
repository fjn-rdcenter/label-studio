import { useEffect } from "react";

export const useRegionsCopyPaste = (entity: any) => {
  useEffect(() => {
    const isFocusable = (el: Node | Window | HTMLElement | null) => {
      if (!el) return false;
      if ((el as Node).nodeType !== Node.ELEMENT_NODE) return false;

      const element = el as HTMLElement;
      const tabIndex = Number.parseInt(element.getAttribute("tabindex") ?? "", 10);
      const isFocusable = element.matches("a, button, input, textarea, select, details, [tabindex], [contenteditable]");

      return isFocusable || tabIndex > -1;
    };

    const allowCopyPaste = () => {
      const selection = window.getSelection();
      const focusNode = selection?.focusNode;
      const nodeIsFocusable = isFocusable(focusNode as HTMLElement);
      const activeElementIsFocusable = isFocusable(document.activeElement);
      const selectionIsCollapsed = selection?.isCollapsed ?? true;

      return selectionIsCollapsed && !nodeIsFocusable && !activeElementIsFocusable;
    };

    const copyToClipboard = (ev: ClipboardEvent) => {
      const { clipboardData } = ev;
      const results = entity.serializedSelection;

      clipboardData?.setData("application/json", JSON.stringify(results));
      ev.preventDefault();
    };

    const retargetToCurrentImage = (results: any[]) => {
      const mountedImages = entity.objects.filter((object: any) => {
        if (object.type !== "image") return false;

        return object.containerRef?.isConnected || object.stageRef?.container()?.isConnected;
      });

      if (mountedImages.length !== 1) return results;

      const targetImage = mountedImages[0];
      const targetControls = entity.toNames.get(targetImage.name) ?? [];
      const mappings = new Map<any, any>();

      for (const result of results) {
        const sourceImage = entity.names.get(result.to_name);

        if (sourceImage?.type !== "image") continue;

        const sourceControl = entity.names.get(result.from_name);
        const resultType = sourceControl?.resultType ?? result.type;
        const sourceControls = (entity.toNames.get(sourceImage.name) ?? []).filter(
          (control: any) => control.resultType === resultType,
        );
        const matchingTargetControls = targetControls.filter((control: any) => control.resultType === resultType);
        const sourceControlIndex = sourceControls.indexOf(sourceControl);
        const targetControl = matchingTargetControls[sourceControlIndex >= 0 ? sourceControlIndex : 0];

        if (!targetControl) return results;
        mappings.set(result, targetControl);
      }

      const imageEntity = targetImage.currentImageEntity;

      return results.map((result: any) => {
        const targetControl = mappings.get(result);

        if (!targetControl) return result;

        const retargeted = {
          ...result,
          from_name: targetControl.name,
          to_name: targetImage.name,
          original_width: imageEntity?.naturalWidth ?? result.original_width,
          original_height: imageEntity?.naturalHeight ?? result.original_height,
          image_rotation: imageEntity?.rotation ?? result.image_rotation,
        };

        if (targetImage.multiImage) retargeted.item_index = targetImage.currentItemIndex;
        else delete retargeted.item_index;

        return retargeted;
      });
    };

    const pasteFromClipboard = (ev: ClipboardEvent) => {
      const { clipboardData } = ev;
      const data = clipboardData?.getData("application/json");

      try {
        const parsedResults = data ? JSON.parse(data) : [];
        const results = retargetToCurrentImage(parsedResults).map((res: any) => {
          return { ...res, readonly: false };
        });

        entity.appendResults(results);
        ev.preventDefault();
      } catch (e) {
        console.error(e);
        return;
      }
    };

    const copyHandler = (ev: Event) => {
      if (!allowCopyPaste()) return;

      copyToClipboard(ev as ClipboardEvent);
    };

    const pasteHandler = (ev: Event) => {
      if (!allowCopyPaste()) return;

      pasteFromClipboard(ev as ClipboardEvent);
    };

    const cutHandler = (ev: Event) => {
      if (!allowCopyPaste()) return;

      copyToClipboard(ev as ClipboardEvent);
      entity.deleteSelectedRegions();
    };

    window.addEventListener("copy", copyHandler);
    window.addEventListener("paste", pasteHandler);
    window.addEventListener("cut", cutHandler);
    return () => {
      window.removeEventListener("copy", copyHandler);
      window.removeEventListener("paste", pasteHandler);
      window.removeEventListener("cut", cutHandler);
    };
  }, [entity.pk ?? entity.id]);
};
